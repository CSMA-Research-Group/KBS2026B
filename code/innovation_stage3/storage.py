"""Atomic immutable records and mutable batch state; no model dependencies."""
import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex[:8])
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def seal(path, value):
    if "record_sha256" in value:
        raise ValueError("Record already sealed")
    atomic(path, {**value, "record_sha256": stable(value)}, immutable=True)


def unseal(path):
    value = read(path)
    expected = value.pop("record_sha256")
    if stable(value) != expected:
        raise ValueError("Changed saved request: " + str(path))
    return value
