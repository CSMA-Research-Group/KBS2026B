"""Append-only scientific outputs and explicit mutable operational status."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'code'))

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)

def rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        yield from (json.loads(line) for line in f)

def save(path, value, mutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not mutable:
        if read(path) == value:
            return
        raise RuntimeError('Refusing to change saved record: ' + str(path))
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    temp.write_bytes(gzip.compress(data, mtime=0) if str(path).endswith('.gz') else data)
    temp.replace(path)

def save_rows(path, values):
    path = Path(path)
    if path.exists():
        raise RuntimeError('Existing analysis file: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(v, sort_keys=True, ensure_ascii=False) + '\n' for v in values)
    path.write_bytes(gzip.compress(text.encode(), mtime=0) if str(path).endswith('.gz') else text.encode())

def table(path, values):
    values = list(values)
    if not values:
        return
    keys = list(dict.fromkeys(k for r in values for k in r))
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(values)

def interval(k, n):
    from scipy.stats import beta
    if n == 0:
        return {'n': 0, 'k': k, 'rate': None, 'cp95_low': None, 'cp95_high': None, 'upper95_one_sided': None}
    return {'n': n, 'k': k, 'rate': k/n,
            'cp95_low': 0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            'cp95_high': 1. if k == n else float(beta.ppf(.975, k+1, n-k)),
            'upper95_one_sided': 1. if k == n else float(beta.ppf(.95, k+1, n-k))}

def summarize(rs):
    n = len(rs)
    x = sum(r['decision'] == 'EXECUTE' for r in rs)
    wrong = sum(bool(r.get('execution_error')) for r in rs)
    solved = [r for r in rs if r['n_feasible'] > 0]
    return {'world_conditions': n, 'executions': x, 'wrong_executions': wrong,
            'correct_executions': sum(bool(r.get('correct_execution')) for r in rs),
            'false_infeasibility': sum(r['decision'] == 'REJECT' and r['n_feasible'] > 0 for r in rs),
            'solvable_conditions': len(solved),
            'clarifications': sum(r['decision'] == 'CLARIFY' for r in rs),
            'rejections': sum(r['decision'] == 'REJECT' for r in rs),
            'abstentions': sum(r['decision'] == 'ABSTAIN' for r in rs)}
