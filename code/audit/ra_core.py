"""Independent analysis core for the research audit (separate from code/revision_20261022).

Design rules
* Pure standard library, so it runs on the project machine and in a clean container.
* Exact rational arithmetic for proposal frequencies, scores, thresholds and envelopes
  (no floating tolerance); this is a deliberate second implementation used for
  differential checks against the frozen float/tolerance implementation.
* Decisions are computed from the set algebra of Theorem 2 (A = L & W, B = U - N),
  not by calling the frozen `decide` loop.
* Read-only with respect to the batch; never writes inside data/.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ['qwen3_8b', 'gemma2_9b']
DOMAINS = ['carrying', 'workflow']
REPS = ['LIST', 'RULE']
DECISIONS = ['EXECUTE', 'CLARIFY', 'REJECT', 'ABSTAIN']
REFERENCE_DECISIONS = ['EXECUTE', 'CLARIFY', 'REJECT']


# ----------------------------------------------------------------------------- I/O

def batch_dir(root: Path = ROOT) -> Path:
    """Batch under audit: $AUDIT_BATCH if set (absolute or relative to ROOT), else the imported snapshot."""
    env = os.environ.get('AUDIT_BATCH')
    if env:
        p = Path(env)
        return p if p.is_absolute() else root / p
    return root / 'work/evidence/main'


def results_dir(experiment: str) -> Path:
    """New analysis directory; never overwrite the saved experiment outputs."""
    base = ROOT / 'work' / 'cost_analysis' / experiment
    run = os.environ.get('AUDIT_RUN')
    return base / run if run else base


def read_json(path):
    path = str(path)
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
        return json.load(f)


def digest(value) -> str:
    """Same canonical JSON digest as the frozen pipeline (common.digest)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


class Batch:
    """Lazy, read-only accessor for one imported batch snapshot."""

    def __init__(self, path: Path | None = None, check_index_dir: Path | None = None):
        self.path = Path(path) if path else batch_dir()
        self.lang = self.path / 'language'
        self.manifest = read_json(self.lang / 'case_manifest.json')
        self.entries = self.manifest['entries']
        self._labels = {}
        self._refs = {}
        self._checks = None
        self.check_index_dir = check_index_dir

    def label(self, cid):
        if cid not in self._labels:
            self._labels[cid] = read_json(self.lang / 'labels' / (cid + '.json.gz'))
        return self._labels[cid]

    def reference(self, cid):
        if cid not in self._refs:
            self._refs[cid] = read_json(self.lang / 'reference' / (cid + '.json.gz'))
        return self._refs[cid]

    def checks(self):
        """{case_id: {goal: (feasible, work_units)}} from the compact index or the raw files."""
        if self._checks is None:
            out = defaultdict(dict)
            if self.check_index_dir is not None:
                for name in sorted(os.listdir(self.check_index_dir)):
                    if not name.endswith('.jsonl.gz'):
                        continue
                    with gzip.open(self.check_index_dir / name, 'rt') as f:
                        for line in f:
                            r = json.loads(line)
                            out[r['case_id']][r['goal']] = (bool(r['feasible']), int(r['work_units']),
                                                            r['public_sha256'])
            else:
                for cid in sorted(os.listdir(self.lang / 'checks')):
                    for name in os.listdir(self.lang / 'checks' / cid):
                        v = read_json(self.lang / 'checks' / cid / name)
                        out[cid][v['goal']] = (bool(v['feasible']), int(v['work_units']), v['public_sha256'])
            self._checks = dict(out)
        return self._checks

    def cases(self, split=None, domain=None, variant=None):
        return [e for e in self.entries if (split is None or e['split'] == split)
                and (domain is None or e['domain'] == domain) and (variant is None or e['variant'] == variant)]

    def jobs(self, model):
        return read_json(self.lang / ('jobs_' + model + '.json'))['jobs']

    def responses(self, model, verify=True):
        """{(case_id, rep, suite, draw, temperature): response} for every saved terminal response."""
        jobs = self.jobs(model)
        out = {}
        rdir = self.lang / model / 'responses'
        present = set(os.listdir(rdir))
        for j in jobs:
            name = j['request_id'] + '.json.gz'
            if name not in present:
                continue
            r = read_json(rdir / name)
            if verify and r['request'] != j:
                raise ValueError('request binding changed: ' + j['request_id'])
            out[(j['case_id'], j['representation'], j['suite'], j['draw'], j['temperature'])] = r
        return out


# ------------------------------------------------------------------- exact scores

def proposal_sets(responses):
    """Candidate set per draw; invalid draws contribute the empty proposal."""
    return [frozenset(r['candidates']) if r.get('valid') else frozenset() for r in responses]


def frequencies(universe, responses):
    sets = proposal_sets(responses)
    k = len(sets)
    if k == 0:
        raise ValueError('no draws')
    return {g: Fraction(sum(g in s for s in sets), k) for g in universe}


def r_sem(p, S, G):
    S = set(S)
    return max([1 - p[g] for g in G if g in S] + [p[g] for g in G if g not in S] + [Fraction(0)])


def r_dec(p, S, P):
    S, P = set(S), set(P)
    return max([1 - p[g] for g in P if g in S] + [p[g] for g in P if g not in S] + [Fraction(0)])


def conformal_rank(n, alpha):
    a = Fraction(str(alpha))
    return math.ceil((n + 1) * (1 - a))


def conformal_q(scores, alpha, clamp=Fraction(1, 2)):
    """Split-conformal order statistic with the paper's lower clamp; q = 1 when rank > n."""
    n = len(scores)
    rank = conformal_rank(n, alpha)
    if rank > n:
        return Fraction(1)
    return max(clamp, sorted(scores)[rank - 1])


def envelope(p, q):
    L = frozenset(g for g, v in p.items() if v > q)
    U = frozenset(g for g, v in p.items() if v >= 1 - q)
    return L, U


# ------------------------------------------------------------- decision algebra

def decide_sets(L, U, W, N, contract='referential'):
    """Theorem 2 decision from A = L & W and B = U - N (returns decision, selected goal)."""
    A = set(L) & set(W)
    B = set(U) - set(N)
    if contract == 'any_valid':
        if A:
            return 'EXECUTE', min(A)
        if not B:
            return 'REJECT', None
        return 'ABSTAIN', None
    if len(A) >= 2:
        return 'CLARIFY', None
    if len(A) == 1 and A == B:
        return 'EXECUTE', next(iter(A))
    if not B:
        return 'REJECT', None
    return 'ABSTAIN', None


def lazy(L, U, feasible_of, contract='referential', order='lower_first', budget=None):
    """Independent lazy evidence acquisition; feasible_of(g) -> (bool, work)."""
    L, U = frozenset(L), frozenset(U)
    assert L <= U
    seq = sorted(U) if order == 'lexical' else sorted(L) + sorted(U - L)
    W, N, queried, work = set(), set(), [], 0
    for g in seq:
        d, sel = decide_sets(L, U, W, N, contract)
        if d != 'ABSTAIN' and order != 'eager':
            break
        if budget is not None and len(queried) >= budget:
            break
        f, w = feasible_of(g)
        queried.append(g)
        work += w
        (W if f else N).add(g)
    d, sel = decide_sets(L, U, W, N, contract)
    return {'decision': d, 'selected': sel, 'queries': len(queried), 'work': work,
            'queried': queried, 'W': sorted(W), 'N': sorted(N)}


def reference_decision(E, contract='referential'):
    if contract == 'any_valid':
        return 'EXECUTE' if E else 'REJECT'
    return 'REJECT' if not E else 'EXECUTE' if len(E) == 1 else 'CLARIFY'


def classify(decision, selected, E, contract='referential'):
    """Outcome category relative to the executable reference set E."""
    E = set(E)
    ref = reference_decision(E, contract)
    cat = None
    if decision == 'ABSTAIN':
        cat = 'abstain'
    elif decision == 'EXECUTE':
        if selected not in E:
            cat = 'invalid_selection'
        elif contract == 'referential' and len(E) >= 2:
            cat = 'false_uniqueness'
        else:
            cat = 'correct_execution'
    elif decision == 'CLARIFY':
        cat = 'correct_clarification' if len(E) >= 2 else 'unsupported_clarification'
    elif decision == 'REJECT':
        cat = 'correct_rejection' if not E else 'false_infeasibility'
    correct = cat in ('correct_execution', 'correct_clarification', 'correct_rejection')
    wrong = cat in ('invalid_selection', 'false_uniqueness', 'unsupported_clarification', 'false_infeasibility')
    return {'reference': ref, 'category': cat, 'correct': correct, 'wrong': wrong,
            'wrong_execution': cat in ('invalid_selection', 'false_uniqueness')}


# ------------------------------------------------------------- exact statistics

def binom_pmf(k, n, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(binom_pmf(i, n, p) for i in range(k + 1))


def _bisect(f, lo=0.0, hi=1.0, it=200):
    # f increasing in p, find root of f(p)=0
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, level=0.95):
    """Two-sided exact interval and one-sided upper bound at `level`."""
    if n == 0:
        return {'k': k, 'n': 0, 'rate': None, 'low': None, 'high': None, 'upper_one_sided': None}
    a = 1 - level
    low = 0.0 if k == 0 else _bisect(lambda p: (1 - binom_cdf(k - 1, n, p)) - a / 2)
    high = 1.0 if k == n else _bisect(lambda p: a / 2 - binom_cdf(k, n, p))
    up1 = 1.0 if k == n else _bisect(lambda p: a - binom_cdf(k, n, p))
    return {'k': k, 'n': n, 'rate': k / n, 'low': low, 'high': high, 'upper_one_sided': up1}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial, p=1/2, doubled smaller tail, capped at 1)."""
    m = b + c
    if m == 0:
        return 1.0
    x = min(b, c)
    return min(1.0, 2 * binom_cdf(x, m, 0.5))


def holm(pvalues):
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [None] * m
    running = 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (m - j) * pvalues[i]))
        adj[i] = running
    return adj


def fmt_frac(x):
    return None if x is None else float(x)
