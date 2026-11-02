"""Read-only local safetensors inventory; no torch, pickle or remote code.

Content hashes identify local bytes, not the authenticity of a publisher.
Fast validation deliberately does not re-read the entire weight payload.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any


MANIFEST_SCHEMA = "ess-local-model-manifest-v1"
REVISION_KIND = "local_file_sha256_manifest_not_official_commit"
MAX_HEADER_BYTES = 100_000_000
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4,
    "U32": 4, "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4,
    "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}
_METADATA_FILES = {
    "config.json", "tokenizer_config.json", "generation_config.json",
    "special_tokens_map.json", "added_tokens.json", "tokenizer.json",
    "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt", "merges.txt",
    "chat_template.jinja",
}
_CONTENT_KEYS = {
    "schema_version", "model_path", "served_model", "model_type", "architectures",
    "weight_format", "trust_remote_code", "local_files_only", "revision_kind", "files",
}


class LocalModelArtifactError(ValueError):
    """Local files do not satisfy the registered safe-loading contract."""


def _unique_object(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise LocalModelArtifactError("Duplicate JSON field in local artifact")
        value[key] = child
    return value


def _json(data: bytes, name: str) -> Any:
    def reject_constant(_):
        raise LocalModelArtifactError("Non-finite JSON value in local artifact")
    try:
        return json.loads(data, object_pairs_hook=_unique_object,
                          parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LocalModelArtifactError(f"Invalid JSON artifact: {name}") from exc


def _directory(path: str | Path) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LocalModelArtifactError("Model directory does not exist or cannot be resolved") from exc
    if not resolved.is_dir():
        raise LocalModelArtifactError("Model path must be a directory")
    return resolved


def _local_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise LocalModelArtifactError("Invalid relative artifact path")
    relative = Path(name)
    if relative.is_absolute() or relative.as_posix() != name or any(part in {".", ".."} for part in relative.parts):
        raise LocalModelArtifactError("Artifact paths must be canonical and remain in the model directory")
    try:
        resolved = (root / relative).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LocalModelArtifactError(f"Missing model artifact: {name}") from exc
    if root not in resolved.parents or not resolved.is_file():
        raise LocalModelArtifactError("Artifact symlink/path escapes the model directory or is not a file")
    if resolved.stat().st_size <= 0:
        raise LocalModelArtifactError(f"Empty model artifact: {name}")
    return resolved


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _sha256_file(path: Path) -> str:
    before = _stamp(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if before != _stamp(path):
        raise LocalModelArtifactError("Artifact changed while its content was being hashed")
    return digest.hexdigest()


def _read_metadata(root: Path, name: str) -> tuple[Any, dict[str, Any]]:
    path = _local_file(root, name)
    before = _stamp(path)
    data = path.read_bytes()
    if before != _stamp(path):
        raise LocalModelArtifactError("Metadata changed while being read")
    value = _json(data, name) if name.endswith(".json") else None
    if name.endswith(".json") and not isinstance(value, dict):
        raise LocalModelArtifactError(f"JSON metadata must be an object: {name}")
    return value, {"kind": "metadata", "size_bytes": len(data),
                   "sha256": hashlib.sha256(data).hexdigest()}


def _safetensors(root: Path, name: str, *, full_hash: bool) -> tuple[set[str], dict[str, Any]]:
    path = _local_file(root, name)
    before = _stamp(path)
    file_size = before[2]
    with path.open("rb") as source:
        length = source.read(8)
        if len(length) != 8:
            raise LocalModelArtifactError("Safetensors file is shorter than its header length field")
        header_size = struct.unpack("<Q", length)[0]
        if not 2 <= header_size <= MAX_HEADER_BYTES or 8 + header_size >= file_size:
            raise LocalModelArtifactError("Safetensors header length/payload length is invalid")
        raw_header = source.read(header_size)
    if len(raw_header) != header_size or not raw_header.startswith(b"{"):
        raise LocalModelArtifactError("Invalid safetensors header bytes")
    header = _json(raw_header, name)
    if not isinstance(header, dict):
        raise LocalModelArtifactError("Safetensors header must be an object")
    payload_size = file_size - 8 - header_size
    tensors, intervals = set(), []
    for tensor, descriptor in header.items():
        if tensor == "__metadata__":
            if not isinstance(descriptor, dict) or any(not isinstance(v, str) for v in descriptor.values()):
                raise LocalModelArtifactError("Invalid safetensors metadata")
            continue
        if not tensor or not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise LocalModelArtifactError("Invalid safetensors tensor descriptor")
        dtype, shape, offsets = descriptor["dtype"], descriptor["shape"], descriptor["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise LocalModelArtifactError("Unsupported safetensors tensor dtype")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise LocalModelArtifactError("Invalid safetensors tensor shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise LocalModelArtifactError("Invalid safetensors tensor offsets")
        start, end = offsets
        if not 0 <= start <= end <= payload_size or end - start != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise LocalModelArtifactError("Safetensors tensor shape/offset/file-length mismatch")
        if end > start:
            intervals.append((start, end))
        tensors.add(tensor)
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise LocalModelArtifactError("Safetensors payload has overlapping tensors or unused holes")
        cursor = end
    if not tensors or cursor != payload_size:
        raise LocalModelArtifactError("Safetensors has no tensor payload or has unaccounted bytes")
    entry = {"kind": "weights", "size_bytes": file_size,
             "header_sha256": hashlib.sha256(length + raw_header).hexdigest(),
             "tensor_count": len(tensors), "tensor_data_bytes": payload_size}
    if full_hash:
        entry["sha256"] = _sha256_file(path)
    if before != _stamp(path):
        raise LocalModelArtifactError("Safetensors file changed during inspection")
    return tensors, entry


def _has_chat_template(config: dict[str, Any]) -> bool:
    template = config.get("chat_template")
    if isinstance(template, str):
        return bool(template.strip())
    if isinstance(template, dict):
        return bool(template) and all(isinstance(v, str) and v.strip() for v in template.values())
    if isinstance(template, list):
        return bool(template) and all(isinstance(v, dict) and isinstance(v.get("name"), str)
                                     and isinstance(v.get("template"), str) and v["template"].strip()
                                     for v in template)
    return False


def inspect_local_model(path: str | Path, full_hash: bool = False) -> dict[str, Any]:
    """Inspect local headers and metadata, optionally hash every weight byte.

    Default mode hashes metadata, but only reads weight headers and stat lengths.
    It proves neither model executability nor a model publisher's authenticity.
    """
    if type(full_hash) is not bool:
        raise LocalModelArtifactError("full_hash must be boolean")
    root = _directory(path)
    files, metadata = {}, {}
    required = {"config.json", "tokenizer_config.json"}
    names = {p.name for p in root.iterdir() if p.name in _METADATA_FILES} | required
    templates = root / "chat_templates"
    if templates.exists():
        if not templates.is_dir():
            raise LocalModelArtifactError("chat_templates must be a directory")
        names.update(p.relative_to(root).as_posix() for p in templates.glob("*.jinja"))
    for name in sorted(names):
        metadata[name], files[name] = _read_metadata(root, name)
    config, tokenizer_config = metadata["config.json"], metadata["tokenizer_config.json"]
    for value in (config, tokenizer_config):
        if value.get("trust_remote_code") or value.get("auto_map"):
            raise LocalModelArtifactError("Remote/custom Python model or tokenizer code is not supported")
    model_type = config.get("model_type")
    architectures = config.get("architectures", [])
    if not isinstance(model_type, str) or not model_type.strip() or not isinstance(architectures, list) or any(not isinstance(x, str) for x in architectures):
        raise LocalModelArtifactError("Invalid model_type or architectures in config.json")
    tokenizer_assets = {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt"}
    if not names & tokenizer_assets and not {"vocab.json", "merges.txt"} <= names:
        raise LocalModelArtifactError("Tokenizer assets are missing")
    external_templates = {name for name in names if name.endswith(".jinja")}
    if not _has_chat_template(tokenizer_config) and not external_templates:
        raise LocalModelArtifactError("A local tokenizer chat_template is required")
    for name in external_templates:
        try:
            nonempty = bool(_local_file(root, name).read_text(encoding="utf-8").strip())
        except UnicodeDecodeError as exc:
            raise LocalModelArtifactError("Chat template must be UTF-8 text") from exc
        if not nonempty:
            raise LocalModelArtifactError("Chat template is empty")
    index_name = "model.safetensors.index.json"
    weight_map = None
    declared_data_size = None
    if (root / index_name).exists():
        index, files[index_name] = _read_metadata(root, index_name)
        files[index_name]["kind"] = "weight_index"
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise LocalModelArtifactError("Safetensors index requires a nonempty weight_map")
        index_metadata = index.get("metadata", {})
        if not isinstance(index_metadata, dict):
            raise LocalModelArtifactError("Safetensors index metadata must be an object")
        if "total_size" in index_metadata:
            declared_data_size = index_metadata["total_size"]
            if type(declared_data_size) is not int or declared_data_size <= 0:
                raise LocalModelArtifactError("Invalid safetensors index total_size")
        for tensor, name in weight_map.items():
            if not tensor or not isinstance(name, str) or Path(name).name != name or "\\" in name or not name.endswith(".safetensors"):
                raise LocalModelArtifactError("Every indexed shard must be directly inside the same model directory")
        weight_names = set(weight_map.values())
    else:
        weight_names = {"model.safetensors"}
    on_disk_weights = {p.name for p in root.glob("*.safetensors")}
    if on_disk_weights != weight_names:
        raise LocalModelArtifactError("Missing, unindexed, or ambiguous safetensors weight files")
    all_tensors = set()
    for name in sorted(weight_names):
        tensors, files[name] = _safetensors(root, name, full_hash=full_hash)
        if all_tensors & tensors:
            raise LocalModelArtifactError("Duplicate tensor names across safetensors shards")
        all_tensors.update(tensors)
        if weight_map is not None and tensors != {tensor for tensor, shard in weight_map.items() if shard == name}:
            raise LocalModelArtifactError("Safetensors index does not match the actual shard tensor headers")
    if declared_data_size is not None and declared_data_size != sum(files[name]["tensor_data_bytes"] for name in weight_names):
        raise LocalModelArtifactError("Safetensors index total_size differs from actual tensor payload bytes")
    resolved_files = [_local_file(root, name) for name in files]
    if len(set(resolved_files)) != len(resolved_files):
        raise LocalModelArtifactError("Multiple artifact names resolve to the same file")
    return {"schema_version": "ess-local-model-inspection-v1", "model_path": str(root),
            "model_type": model_type, "architectures": architectures,
            "weight_format": "safetensors", "trust_remote_code": False, "local_files_only": True,
            "files": files, "weight_files": sorted(weight_names), "index_file": index_name if weight_map is not None else None,
            "full_weights_verified": full_hash,
            "verification_level": "full_file_sha256" if full_hash else "metadata_sha256_and_weight_headers_lengths"}


def _content_hash(content: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise LocalModelArtifactError("Manifest content is not canonical finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _served_name(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+@-]*", value):
        raise LocalModelArtifactError("served_model must be a bounded exact identifier using the local server's supported characters")
    return value


def build_model_manifest(model_path: str | Path, served_model: str) -> dict[str, Any]:
    """Explicit full-byte registration; this is the expensive one-time operation."""
    served_model = _served_name(served_model)
    inspected = inspect_local_model(model_path, full_hash=True)
    content = {key: inspected[key] for key in ("model_path", "model_type", "architectures", "weight_format",
                                              "trust_remote_code", "local_files_only", "files")}
    content.update(schema_version=MANIFEST_SCHEMA, served_model=served_model, revision_kind=REVISION_KIND)
    digest = _content_hash(content)
    return {**content, "content_manifest_sha256": digest, "model_revision": "local-sha256:" + digest}


def validate_model_manifest(manifest: dict[str, Any], model_path: str | Path,
                            verify_full_hash: bool = False) -> dict[str, Any]:
    """Validate local registration; quick mode cannot detect same-size payload edits."""
    if type(verify_full_hash) is not bool or not isinstance(manifest, dict):
        raise LocalModelArtifactError("Invalid manifest or full-hash verification flag")
    if set(manifest) != _CONTENT_KEYS | {"content_manifest_sha256", "model_revision"}:
        raise LocalModelArtifactError("Manifest fields do not match the local-model schema")
    if manifest["schema_version"] != MANIFEST_SCHEMA or manifest["revision_kind"] != REVISION_KIND:
        raise LocalModelArtifactError("Unsupported local-model manifest schema/revision kind")
    if manifest["weight_format"] != "safetensors" or manifest["trust_remote_code"] is not False or manifest["local_files_only"] is not True:
        raise LocalModelArtifactError("Manifest does not enforce safe local-only safetensors loading")
    _served_name(manifest["served_model"])
    root = _directory(model_path)
    if manifest["model_path"] != str(root):
        raise LocalModelArtifactError("Manifest is bound to a different resolved model directory")
    digest = _content_hash({key: manifest[key] for key in _CONTENT_KEYS})
    if manifest["content_manifest_sha256"] != digest or manifest["model_revision"] != "local-sha256:" + digest:
        raise LocalModelArtifactError("Manifest content hash or local revision does not match")
    registered = manifest["files"]
    if not isinstance(registered, dict) or not registered:
        raise LocalModelArtifactError("Manifest file inventory is missing")
    for name, entry in registered.items():
        _local_file(root, name)
        if not isinstance(entry, dict) or type(entry.get("size_bytes")) is not int or entry["size_bytes"] <= 0:
            raise LocalModelArtifactError("Invalid manifest file size")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise LocalModelArtifactError("Every registered file requires a SHA256 hash")
        if entry.get("kind") == "weights":
            if any(type(entry.get(key)) is not int or entry[key] <= 0 for key in ("tensor_count", "tensor_data_bytes")):
                raise LocalModelArtifactError("Invalid registered safetensors shape/size metadata")
            if not isinstance(entry.get("header_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["header_sha256"]):
                raise LocalModelArtifactError("Missing or invalid safetensors header hash")
    inspected = inspect_local_model(root, full_hash=verify_full_hash)
    if set(inspected["files"]) != set(registered):
        raise LocalModelArtifactError("Current model file inventory differs from the registered manifest")
    for key in ("model_type", "architectures"):
        if inspected[key] != manifest[key]:
            raise LocalModelArtifactError("Model configuration differs from the registered manifest")
    for name, actual in inspected["files"].items():
        expected = registered[name]
        required_keys = set(actual) | {"sha256"}
        if set(expected) != required_keys:
            raise LocalModelArtifactError("Invalid registered artifact entry fields")
        if any(expected[key] != value for key, value in actual.items()):
            raise LocalModelArtifactError(f"Registered artifact no longer matches local bytes: {name}")
    return {"status": "VALID", "model_path": str(root), "served_model": manifest["served_model"],
            "content_manifest_sha256": digest, "model_revision": manifest["model_revision"],
            "revision_kind": REVISION_KIND, "full_weights_verified": verify_full_hash,
            "verification_level": inspected["verification_level"],
            "official_origin_verified": False}
