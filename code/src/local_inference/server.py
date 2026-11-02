"""Loopback-only, authenticated, non-streaming local Chat Completions server.

No model or ML dependency is imported until TransformersChatEngine is created.
Relative configuration/input/output paths resolve against the project root, not
the current working directory. This is a research service, not a public API.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
import uuid

from src.utils.io import file_sha256, stable_hash, utc_now, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FIELDS = {
    "model_path", "served_model", "api_key_env", "port", "device", "dtype",
    "max_input_tokens", "max_output_tokens", "model_revision", "artifact_manifest",
}
REQUEST_FIELDS = {"model", "messages", "temperature", "max_tokens", "seed", "top_p", "stream"}
TEMPLATE_POLICIES = {
    "qwen2": "native_chat_template_v1",
    "gemma2": "native_chat_template_system_prefix_into_first_user_v1",
}
SYSTEM_PREFIX = "[System instructions]\n"
USER_PREFIX = "\n\n[User message]\n"
GENERATION_FIXED = {
    "num_beams": 1, "num_beam_groups": 1, "num_return_sequences": 1,
    "min_length": 0, "min_new_tokens": 0, "early_stopping": False,
    "max_time": None, "stop_strings": None, "penalty_alpha": None,
    "dola_layers": None, "use_cache": True, "cache_config": None,
    "min_p": None, "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
    "diversity_penalty": 0.0, "repetition_penalty": 1.0,
    "encoder_repetition_penalty": 1.0, "length_penalty": 1.0,
    "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0,
    "bad_words_ids": None, "force_words_ids": None, "constraints": None,
    "forced_bos_token_id": None, "forced_eos_token_id": None,
    "renormalize_logits": False, "remove_invalid_values": False,
    "exponential_decay_length_penalty": None, "suppress_tokens": None,
    "begin_suppress_tokens": None, "sequence_bias": None, "token_healing": False,
    "guidance_scale": None, "watermarking_config": None,
    "output_attentions": False, "output_hidden_states": False,
    "output_scores": False, "output_logits": False, "return_dict_in_generate": False,
}


class LocalServerConfigurationError(ValueError):
    pass


class RequestError(ValueError):
    def __init__(self, message: str, *, param: str | None = None, status: int = 400,
                 code: str = "invalid_request_error"):
        super().__init__(message)
        self.param, self.status, self.code = param, status, code


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def _strict_json(raw: str | bytes):
    def invalid_constant(_):
        raise ValueError("Non-finite JSON number")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def _project_path(value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise LocalServerConfigurationError(f"{field} must be a nonempty filesystem path")
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _integer(value, minimum, maximum):
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def validate_server_config(config: Any) -> dict[str, Any]:
    """Validate settings without importing ML libraries or reading model weights."""
    if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
        raise LocalServerConfigurationError("Config must contain exactly the documented server fields")
    result = copy.deepcopy(config)
    for name in ("served_model", "model_revision"):
        if not isinstance(result[name], str) or not result[name] or len(result[name]) > 256:
            raise LocalServerConfigurationError(f"{name} must be a nonempty bounded string")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+@-]*", result[name]):
            raise LocalServerConfigurationError(f"{name} contains unsupported characters")
    if not isinstance(result["api_key_env"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result["api_key_env"]):
        raise LocalServerConfigurationError("api_key_env must name an environment variable")
    if not _integer(result["port"], 0, 65535):
        raise LocalServerConfigurationError("port must be an integer in [0, 65535]; 0 requests an ephemeral test port")
    if result["device"] != "cuda:0":
        raise LocalServerConfigurationError("Only the single visible GPU cuda:0 is supported")
    if not isinstance(result["dtype"], str) or result["dtype"] not in ("float16", "float32"):
        raise LocalServerConfigurationError("dtype must be explicitly float16 or float32; there is no precision fallback")
    for name in ("max_input_tokens", "max_output_tokens"):
        if not _integer(result[name], 1, 131072):
            raise LocalServerConfigurationError(f"{name} must be an integer in [1, 131072]")
    for name in ("model_path", "artifact_manifest"):
        result[name] = str(_project_path(result[name], field=name))
    return result


def load_server_config(path: str | Path) -> dict[str, Any]:
    selected = _project_path(str(path), field="config")
    try:
        return validate_server_config(_strict_json(selected.read_bytes()))
    except (OSError, ValueError) as exc:
        raise LocalServerConfigurationError("Cannot read a valid local server configuration") from exc


def _read_api_key(config):
    value = os.environ.get(config["api_key_env"], "")
    if not value or not re.fullmatch(r"[\x21-\x7e]+", value):
        raise LocalServerConfigurationError("api_key_env must contain a nonempty, header-safe Bearer credential")
    return value


def validate_chat_request(body: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Reject unknown/ignored parameters and preserve all accepted message text."""
    if not isinstance(body, dict):
        raise RequestError("Request body must be a JSON object")
    if set(body) - REQUEST_FIELDS:
        raise RequestError("Unsupported request parameter", param=sorted(set(body) - REQUEST_FIELDS)[0])
    if body.get("model") != config["served_model"]:
        raise RequestError("Requested model is not served here", param="model", status=404, code="model_not_found")
    if "stream" in body and body["stream"] is not False:
        raise RequestError("Only stream=false is supported", param="stream")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 256:
        raise RequestError("messages must be a nonempty text chat with at most 256 messages", param="messages")
    next_role = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise RequestError("Each message must have exactly role and text content", param="messages")
        if not isinstance(message["content"], str) or not message["content"]:
            raise RequestError("Message content must be nonempty text", param="messages")
        role = message["role"]
        if index == 0 and role == "system":
            continue
        if role != next_role:
            raise RequestError("One initial system message followed by alternating user/assistant turns is required", param="messages")
        next_role = "assistant" if role == "user" else "user"
    if messages[-1]["role"] != "user":
        raise RequestError("The final message must be a user turn", param="messages")
    temperature = body.get("temperature", 0.0)
    top_p = body.get("top_p", 1.0)
    for name, value, low, high in (("temperature", temperature, 0, 2), ("top_p", top_p, 0, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise RequestError(f"{name} must be a finite number in [{low}, {high}]", param=name)
    if top_p == 0:
        raise RequestError("top_p must be greater than zero", param="top_p")
    if temperature == 0 and top_p != 1:
        raise RequestError("Greedy temperature=0 requires top_p=1; top_p would otherwise be ignored", param="top_p")
    max_tokens = body.get("max_tokens", config["max_output_tokens"])
    if not _integer(max_tokens, 1, config["max_output_tokens"]):
        raise RequestError("max_tokens exceeds the configured positive output limit", param="max_tokens")
    seed = body.get("seed", 0)
    if not _integer(seed, 0, 2**63 - 1):
        raise RequestError("seed must be an integer in [0, 2**63-1]", param="seed")
    return {"messages": copy.deepcopy(messages), "temperature": float(temperature),
            "top_p": float(top_p), "max_tokens": max_tokens, "seed": seed}


def prepare_template_messages(messages: list[dict[str, str]], model_type: str):
    """Gemma's unsupported system role is an explicitly tagged first-user prefix."""
    if model_type not in TEMPLATE_POLICIES:
        raise LocalServerConfigurationError("No declared chat-template policy for this architecture")
    result = copy.deepcopy(messages)
    if model_type == "gemma2" and result[0]["role"] == "system":
        system = result.pop(0)["content"]
        result[0]["content"] = SYSTEM_PREFIX + system + USER_PREFIX + result[0]["content"]
    return result


def _check_context(prompt_tokens, output_tokens, config, context_limit):
    if prompt_tokens > config["max_input_tokens"]:
        raise RequestError("Tokenized input exceeds max_input_tokens; input was not truncated", param="messages", code="context_length_exceeded")
    if prompt_tokens + output_tokens > context_limit:
        raise RequestError("Input plus requested output exceeds the model context; no tokens were truncated", param="max_tokens", code="context_length_exceeded")


def classify_completion(token_ids: list[int], eos_ids: list[int], max_tokens: int) -> str:
    if not token_ids or len(token_ids) > max_tokens:
        raise RuntimeError("Engine returned an invalid generated token count")
    if any(token in eos_ids for token in token_ids[:-1]):
        raise RuntimeError("Engine generated tokens after a declared EOS")
    if token_ids[-1] in eos_ids:
        return "stop"
    if len(token_ids) == max_tokens:
        return "length"
    raise RuntimeError("Generation stopped without EOS or the declared length limit")


def validate_registered_artifact(config):
    from src.local_inference.artifacts import validate_model_manifest
    path = Path(config["artifact_manifest"])
    try:
        raw = path.read_bytes()
        manifest = _strict_json(raw)
        report = validate_model_manifest(manifest, config["model_path"], verify_full_hash=False)
    except (OSError, ValueError) as exc:
        raise LocalServerConfigurationError("Local model artifact registration validation failed") from exc
    if manifest.get("served_model") != config["served_model"] or manifest.get("model_revision") != config["model_revision"]:
        raise LocalServerConfigurationError("Config model identity/revision does not match the frozen artifact manifest")
    if report.get("full_weights_verified") is not False:
        raise LocalServerConfigurationError("Artifact validation did not declare its startup verification scope")
    return {"manifest": manifest, "validation": report,
            "artifact_manifest_sha256": hashlib.sha256(raw).hexdigest()}


class TransformersChatEngine:
    """A single-GPU engine; constructing this object performs the actual load."""

    def __init__(self, config: dict[str, Any], artifact_validation: dict[str, Any]):
        self.config = validate_server_config(config)
        registration = artifact_validation["manifest"]
        self.model_type = registration["model_type"]
        allowed = {"qwen2": ["Qwen2ForCausalLM"], "gemma2": ["Gemma2ForCausalLM"]}
        if registration.get("architectures") != allowed.get(self.model_type):
            raise LocalServerConfigurationError("Only native Qwen2 and Gemma2 causal text models are supported")
        model_path = Path(self.config["model_path"])
        model_json = _strict_json((model_path / "config.json").read_bytes())
        if model_json.get("quantization_config"):
            raise LocalServerConfigurationError("Quantized weights are not part of this explicit-precision engine contract")
        self.context_limit = model_json.get("max_position_embeddings")
        if not _integer(self.context_limit, 1, 2**31 - 1):
            raise LocalServerConfigurationError("A finite native model context limit is required")
        if self.config["max_input_tokens"] + self.config["max_output_tokens"] > self.context_limit:
            raise LocalServerConfigurationError("Configured input plus output limits exceed native model context")
        # Lazy imports: tests and metadata inspection do not initialize CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
            raise LocalServerConfigurationError("Unsupported CUBLAS_WORKSPACE_CONFIG for deterministic execution")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch, self.GenerationConfig = torch, GenerationConfig
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise LocalServerConfigurationError("A visible CUDA GPU is required; there is no CPU/offload fallback")
        self.device = torch.device(self.config["device"])
        selected_dtype = {"float16": torch.float16, "float32": torch.float32}[self.config["dtype"]]
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        if not isinstance(self.tokenizer.chat_template, str) or not self.tokenizer.chat_template:
            raise LocalServerConfigurationError("A single pinned native tokenizer chat_template is required")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=selected_dtype,
            low_cpu_mem_usage=False, attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        if self.model.config.model_type != self.model_type:
            raise LocalServerConfigurationError("Loaded architecture does not match the registered model")
        parameters = list(self.model.parameters())
        if not parameters or any(parameter.numel() == 0 or parameter.device != self.device
                                 or parameter.dtype != selected_dtype for parameter in parameters):
            raise LocalServerConfigurationError(
                "Loaded parameters must be nonempty and match the explicitly selected dtype and GPU")
        # Publish the verified parameter contract, not an unverified config label.
        self.parameter_dtype = "float16" if parameters[0].dtype == torch.float16 else "float32"
        self.parameter_device = str(parameters[0].device)
        generation_path = model_path / "generation_config.json"
        special = _strict_json(generation_path.read_bytes()) if generation_path.is_file() else {}
        eos = special.get("eos_token_id", model_json.get("eos_token_id", self.tokenizer.eos_token_id))
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        if not self.eos_ids or any(not _integer(value, 0, 2**31 - 1) for value in self.eos_ids):
            raise LocalServerConfigurationError("Pinned EOS token IDs are required")
        pad = special.get("pad_token_id", model_json.get("pad_token_id", self.tokenizer.pad_token_id))
        if pad is None:
            pad = self.eos_ids[0]
        bos = special.get("bos_token_id", model_json.get("bos_token_id", self.tokenizer.bos_token_id))
        if not _integer(pad, 0, 2**31 - 1) or (bos is not None and not _integer(bos, 0, 2**31 - 1)):
            raise LocalServerConfigurationError("Pinned special token IDs are invalid")
        self.generation_base = {**GENERATION_FIXED, "do_sample": False,
            "temperature": 1.0, "top_p": 1.0, "top_k": None,
            "eos_token_id": self.eos_ids, "pad_token_id": pad, "bos_token_id": bos,
            "cache_implementation": "hybrid" if self.model_type == "gemma2" else None}
        # Never inherit checkpoint-specific top-k, repetition penalties, or stops.
        self.model.generation_config = GenerationConfig(**self.generation_base)
        self._lock = threading.Lock()
        torch.cuda.synchronize(self.device)
        self.metadata = {
            "engine": "transformers_local_explicit_precision_v2", "test_only": False,
            "model_path": str(model_path), "model_type": self.model_type,
            "served_model": self.config["served_model"], "model_revision": self.config["model_revision"],
            "artifact_manifest_sha256": artifact_validation["artifact_manifest_sha256"],
            "content_manifest_sha256": registration["content_manifest_sha256"],
            "artifact_validation": artifact_validation["validation"],
            "local_files_only": True, "trust_remote_code": False,
            "weight_format": "safetensors", "dtype": self.parameter_dtype, "device": self.parameter_device,
            "device_name": torch.cuda.get_device_name(self.device),
            "device_capability": list(torch.cuda.get_device_capability(self.device)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "safetensors")},
            "python_version": sys.version.split()[0], "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "attention_implementation": "eager",
            "deterministic_algorithms": True, "tf32": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "seed_policy": "per_request_cpu_and_selected_cuda_rng_fork_v1",
            "seed_reproducibility_scope": "same_frozen_artifacts_software_hardware_not_cross_version_guarantee",
            "template_policy": TEMPLATE_POLICIES[self.model_type],
            "chat_template_sha256": hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
            "system_prefix": SYSTEM_PREFIX if self.model_type == "gemma2" else None,
            "user_prefix": USER_PREFIX if self.model_type == "gemma2" else None,
            "generation_base": self.generation_base,
            "generation_defaults": {"temperature": 0.0, "top_p": 1.0, "seed": 0,
                                    "max_tokens": self.config["max_output_tokens"]},
            "sampling_top_k": 0, "native_context_limit": self.context_limit,
            "max_input_tokens": self.config["max_input_tokens"],
            "max_output_tokens": self.config["max_output_tokens"],
            "input_truncation": False, "load_seconds": time.perf_counter() - started,
        }

    def generate(self, messages, temperature, max_tokens, seed, top_p):
        request = validate_chat_request({"model": self.config["served_model"], "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "seed": seed, "top_p": top_p}, self.config)
        waiting = time.perf_counter()
        with self._lock:
            started = time.perf_counter()
            template_messages = prepare_template_messages(request["messages"], self.model_type)
            encoded = self.tokenizer.apply_chat_template(
                template_messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, truncation=False,
            )
            prompt_tokens = int(encoded["input_ids"].shape[1])
            _check_context(prompt_tokens, max_tokens, self.config, self.context_limit)
            prompt_ids = encoded["input_ids"][0].tolist()
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            generation_values = {**self.generation_base, "max_new_tokens": max_tokens,
                "do_sample": temperature > 0, "temperature": temperature if temperature > 0 else 1.0,
                "top_p": top_p, "top_k": 0 if temperature > 0 else None}
            generation = self.GenerationConfig(**generation_values)
            torch = self.torch
            with torch.random.fork_rng(devices=[self.device.index]), torch.cuda.device(self.device), torch.inference_mode():
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.synchronize(self.device)
                generation_started = time.perf_counter()
                output = self.model.generate(**encoded, generation_config=generation)
                torch.cuda.synchronize(self.device)
                generation_seconds = time.perf_counter() - generation_started
            token_ids = output[0, prompt_tokens:].tolist()
            finish = classify_completion(token_ids, self.eos_ids, max_tokens)
            content = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return {"content": content, "prompt_tokens": prompt_tokens,
                "completion_tokens": len(token_ids), "finish_reason": finish,
                "runtime_metadata": {"seed": seed, "temperature": temperature, "top_p": top_p,
                    "dtype": self.parameter_dtype, "device": self.parameter_device,
                    "max_tokens": max_tokens, "do_sample": temperature > 0,
                    "generation_config_sha256": stable_hash(generation.to_dict()),
                    "prompt_token_ids_sha256": stable_hash(prompt_ids),
                    "completion_token_ids_sha256": stable_hash(token_ids),
                    "template_policy": TEMPLATE_POLICIES[self.model_type],
                    "input_truncated": False, "completion_tokens_include_eos": True,
                    "queue_seconds": started - waiting, "generation_seconds": generation_seconds,
                    "engine_seconds": time.perf_counter() - started}}


def _completion_response(result, request, config, fingerprint, test_only):
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        raise RuntimeError("Invalid engine response")
    for name in ("prompt_tokens", "completion_tokens"):
        if not _integer(result.get(name), 1, 2**31 - 1):
            raise RuntimeError("Engine did not supply actual token counts")
    if result["completion_tokens"] > request["max_tokens"] or result["prompt_tokens"] > config["max_input_tokens"]:
        raise RuntimeError("Engine exceeded declared token limits")
    if result.get("finish_reason") not in {"stop", "length"}:
        raise RuntimeError("Engine did not supply an explicit finish reason")
    if result["finish_reason"] == "length" and result["completion_tokens"] != request["max_tokens"]:
        raise RuntimeError("Length finish reason must consume the declared token limit")
    metadata = result.get("runtime_metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError("Invalid engine runtime metadata")
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
        "created": int(time.time()), "model": config["served_model"],
        "model_revision": config["model_revision"], "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]},
                     "finish_reason": result["finish_reason"]}],
        "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                  "total_tokens": result["prompt_tokens"] + result["completion_tokens"]},
        "ess_runtime": {**metadata, "seed": request["seed"], "temperature": request["temperature"],
                        "top_p": request["top_p"], "max_tokens": request["max_tokens"],
                        "test_only": test_only, "service_manifest_sha256": fingerprint}}


class _LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        # Never print handler exceptions containing request text or credentials.
        pass


def create_http_server(config: dict[str, Any], engine, *, service_manifest=None):
    """Create (do not run) a loopback server. Fake engines must declare test_only."""
    config = validate_server_config(config)
    key = _read_api_key(config)
    engine_metadata = copy.deepcopy(getattr(engine, "metadata", {}))
    if type(engine_metadata.get("test_only")) is not bool:
        raise LocalServerConfigurationError("Engine metadata must explicitly declare test_only")
    manifest = copy.deepcopy(service_manifest or {"schema_version": "ess-local-service-v1", "engine": engine_metadata})
    fingerprint = stable_hash(manifest)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ESSLocal/1"
        sys_version = ""

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *_):
            pass

        def _json(self, status, body):
            raw = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("X-ESS-Model-Revision", config["model_revision"])
            self.end_headers()
            self.wfile.write(raw)
            self.close_connection = True

        def _error(self, error):
            self._json(error.status, {"error": {"message": str(error), "type": error.code,
                       "code": error.code, "param": error.param}})

        def _authorized(self):
            supplied = self.headers.get_all("Authorization", [])
            if len(supplied) != 1 or not hmac.compare_digest(supplied[0].encode("utf-8"), ("Bearer " + key).encode("ascii")):
                self._error(RequestError("Valid Bearer authentication is required", status=401, code="authentication_error"))
                return False
            return True

        def do_GET(self):
            if not self._authorized():
                return
            public = {"model_revision": config["model_revision"], "system_fingerprint": fingerprint,
                      "test_only": engine_metadata["test_only"]}
            if self.path == "/health":
                self._json(200, {"status": "ready", "served_model": config["served_model"], **public})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": config["served_model"],
                    "object": "model", "owned_by": "local", **public}]})
            else:
                self._error(RequestError("Unknown endpoint", status=404, code="not_found"))

        def do_POST(self):
            if not self._authorized():
                return
            if self.path != "/v1/chat/completions":
                self._error(RequestError("Unknown endpoint", status=404, code="not_found"))
                return
            try:
                if self.headers.get_all("Transfer-Encoding"):
                    raise RequestError("Transfer-Encoding is unsupported")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
                    raise RequestError("Exactly one valid Content-Length is required")
                size = int(lengths[0])
                if not 1 <= size <= 2 * 1024 * 1024:
                    raise RequestError("Request body exceeds the 2 MiB transport limit", status=413)
                if len(self.headers.get_all("Content-Type", [])) != 1 or self.headers.get_content_type() != "application/json":
                    raise RequestError("Content-Type must be application/json", status=415)
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise RequestError("Incomplete request body")
                try:
                    body = _strict_json(raw)
                except (UnicodeError, ValueError):
                    raise RequestError("Request must be strict UTF-8 JSON with unique fields") from None
                request = validate_chat_request(body, config)
                queued = time.perf_counter()
                with lock:
                    entered = time.perf_counter()
                    result = engine.generate(**request)
                if isinstance(result, dict) and isinstance(result.get("runtime_metadata"), dict):
                    result["runtime_metadata"]["server_queue_seconds"] = entered - queued
                self._json(200, _completion_response(result, request, config, fingerprint, engine_metadata["test_only"]))
            except RequestError as exc:
                self._error(exc)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception:
                # No traceback, exception string, prompt, or authorization log.
                self._error(RequestError("Local generation failed; no successful completion was produced",
                                         status=500, code="local_generation_error"))

        def do_OPTIONS(self):
            self._error(RequestError("This local service does not allow browser CORS", status=405))

    server = _LocalHTTPServer(("127.0.0.1", config["port"]), Handler)
    server.service_fingerprint = fingerprint
    return server


def _reserve_manifest_directory(value: str | Path) -> Path:
    output = _project_path(str(value), field="manifest-output")
    allowed = (PROJECT_ROOT / "data" / "results").resolve()
    if output == allowed or allowed not in output.parents:
        raise LocalServerConfigurationError("manifest-output must be a new directory below project data/results")
    output.mkdir(parents=True, exist_ok=False)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-output", required=True,
                        help="A NEW directory below project data/results (never overwrite)")
    args = parser.parse_args(argv)
    output = None
    server = None
    try:
        config = load_server_config(args.config)
        if config["port"] == 0:
            raise LocalServerConfigurationError("port=0 is test-only; production config requires a fixed port")
        _read_api_key(config)
        artifact = validate_registered_artifact(config)
        output = _reserve_manifest_directory(args.manifest_output)
        write_json(output / "service_state.json", {"status": "LOADING", "created_at": utc_now()})
        engine = TransformersChatEngine(config, artifact)
        manifest = {"schema_version": "ess-local-service-v1", "created_at": utc_now(),
            "status": "LOADED", "config": config, "config_sha256": stable_hash(config),
            "source_sha256": file_sha256(Path(__file__)), "engine": engine.metadata,
            "transport": {"host": "127.0.0.1", "port": config["port"], "tls": False,
                "authentication": "Bearer_from_environment", "request_logging": False,
                "streaming": False, "max_body_bytes": 2 * 1024 * 1024}}
        server = create_http_server(config, engine, service_manifest=manifest)
        write_json(output / "service_manifest.json", manifest)
        write_json(output / "service_state.json", {"status": "READY", "updated_at": utc_now(),
            "service_manifest_sha256": server.service_fingerprint})
        print(f"Local service ready on 127.0.0.1:{config['port']}; request logging disabled", flush=True)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        if output:
            write_json(output / "service_state.json", {"status": "STOPPED", "updated_at": utc_now()})
        return 0
    except Exception as exc:
        if output:
            write_json(output / "service_state.json", {"status": "FAILED", "updated_at": utc_now(),
                "error_category": type(exc).__name__, "error_detail_omitted": True})
        print(f"Local service failed ({type(exc).__name__}); inspect configuration, artifact registration, and dependency availability. No model completion was produced.", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
