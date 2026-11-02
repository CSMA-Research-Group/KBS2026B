"""Shared file and resource helpers for local project maintenance."""
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "work/evidence"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(relative, root=ROOT):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a project-relative path: {relative}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves the project: {relative}")
    return target


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def new_run(kind):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "work" / kind / stamp
    out.mkdir(parents=True, exist_ok=False)
    return out


def bounded_map(pool, function, items, chunk_size=128):
    """Limit queued futures when scanning hundreds of thousands of small files."""
    iterator = iter(items)
    while chunk := list(islice(iterator, chunk_size)):
        yield from pool.map(function, chunk)


def io_resources():
    if hasattr(os, "sched_getaffinity"):
        sys.path.insert(0, str(ROOT / "code"))
        from experiment_resources import plan_resources
        plan = plan_resources(4)
        plan["workers"] = min(4, plan["workers"])
        return plan
    return {
        "mode": "local_file_audit", "platform": sys.platform,
        "workers": min(4, max(1, (os.cpu_count() or 2) - 1)),
        "gpu_requests": 0, "timing_results": "not_measured",
        "note": "Bounded I/O on hosts without Linux affinity and cgroup interfaces.",
    }


def current_batch():
    from archives import prepare
    prepare(['cpu'])
    return EVIDENCE / 'main', {'status':'COMPLETE', 'snapshot':'2026-10-22'}
