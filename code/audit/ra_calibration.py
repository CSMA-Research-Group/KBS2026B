"""Calibration procedures used by the audit (independent of the frozen implementation).

* split-conformal thresholds for r_sem / r_dec (exact rationals)  -> frozen-paper policies
* DECISION_RISK_CRC: conformal risk control on the full-evidence decision loss
* KNOWNO_STYLE_UPPER_ONLY: upper-envelope coverage of E only (baseline adaptation)
* ENSEMBLE_LR records: frequencies pooled over LIST and RULE draws
The packaged scope is defined in docs/工作计划.md.
"""
from __future__ import annotations

from fractions import Fraction as F

import ra_core as rc

HALF = F(1, 2)


def world_record(b, resp, cid, reps=('LIST',), k=3, suite='primary', temp=0.7):
    """Exact per-world record; returns None if any requested draw is missing."""
    lab = b.label(cid)
    G = lab['universe']
    S = frozenset(lab['semantic'])
    P = frozenset(b.reference(cid)['feasible'])
    rs = []
    for rep in reps:
        for d in range(k):
            key = (cid, rep, suite, d, temp)
            if key not in resp:
                return None
            rs.append(resp[key])
    p = rc.frequencies(G, rs)
    sets = rc.proposal_sets(rs)
    return {'case_id': cid, 'G': tuple(G), 'S': S, 'P': P, 'E': S & P, 'p': p, 'k': len(rs),
            'union': frozenset().union(*sets), 'valid_any': any(r['valid'] for r in rs),
            'n_invalid': sum(not r['valid'] for r in rs),
            'output_tokens': sum(int(r.get('output_tokens') or 0) for r in rs),
            'input_tokens': sum(int(r.get('input_tokens') or 0) for r in rs)}


def q_grid(k):
    vals = {HALF, F(1)} | {F(j, k) for j in range(k + 1) if F(j, k) >= HALF}
    return sorted(vals)


def env_q(rec, q):
    return rc.envelope(rec['p'], q)


def chain5(rec, level):
    """k=3 monotone chain of (L, U) envelopes, level 0 (most aggressive) .. 4 (L=empty, U=G)."""
    p = rec['p']
    ge = lambda t: frozenset(g for g, v in p.items() if v >= t)
    third, two = F(1, 3), F(2, 3)
    table = [(ge(two), ge(two)), (ge(F(1)), ge(two)), (ge(F(1)), ge(third)), (frozenset(), ge(third)),
             (frozenset(), frozenset(p))]
    return table[level]


def full_decision(L, U, P, contract='referential'):
    U = frozenset(U)
    return rc.decide_sets(L, U, U & P, U - P, contract)


def full_loss(rec, L, U, contract='referential'):
    if not rec['valid_any']:
        return 0
    d, sel = full_decision(L, U, rec['P'], contract)
    return int(rc.classify(d, sel, rec['E'], contract)['wrong'])


def conformal_threshold(recs, kind, alpha):
    if kind == 'semantic':
        scores = [rc.r_sem(r['p'], r['S'], r['G']) for r in recs]
    elif kind == 'decision':
        scores = [rc.r_dec(r['p'], r['S'], r['P']) for r in recs]
    elif kind == 'knowno':
        scores = [max([1 - r['p'][g] for g in r['E']] + [F(0)]) for r in recs]
    else:
        raise ValueError(kind)
    return rc.conformal_q(scores, alpha)


def crc_select(recs, alpha, levels, envelope_of, contract='referential'):
    """Smallest level with (sum loss + 1)/(n+1) <= alpha; asserts monotone empirical risk."""
    n = len(recs)
    a = F(str(alpha))
    risks = []
    for lev in levels:
        risks.append(sum(full_loss(r, *envelope_of(r, lev), contract) for r in recs))
    for x, y in zip(risks, risks[1:]):
        if y > x:
            raise AssertionError('non-monotone empirical decision risk; CRC premise violated')
    if risks[-1] != 0:
        raise AssertionError('most conservative level has nonzero loss; checker unsound?')
    for lev, rk in zip(levels, risks):
        if F(rk + 1, n + 1) <= a:
            return lev, risks
    return levels[-1], risks


def knowno_decision(rec, q):
    """Upper-only envelope with full checking: execute singleton, reject empty, else ask (abstain)."""
    if not rec['valid_any']:
        return 'ABSTAIN', None
    U = frozenset(g for g, v in rec['p'].items() if v >= 1 - q)
    feas = U & rec['P']
    if len(feas) == 1:
        return 'EXECUTE', next(iter(feas))
    if not feas:
        return 'REJECT', None
    return 'ABSTAIN', None
