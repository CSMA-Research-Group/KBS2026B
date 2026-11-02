"""Audit proposal: lazy evidence acquisition with dead-end (certain-abstention) stopping.

The paper's lazy procedure stops when a non-abstaining decision is determined, but it keeps
querying when the final decision can only be ABSTAIN.  Given W, N and the unresolved part
R = U - W - N, a final decision is still reachable only if
  CLARIFY     |L & (W | R)| >= 2                          (referential)
  EXECUTE(g)  g in L & (W | R) and W <= {g}               (referential)
              L & (W | R) nonempty                        (any-valid)
  REJECT      W is empty
If none is reachable, the outcome of full-budget lazy checking is ABSTAIN for certain, so the
procedure may stop and abstain.  Decisions are unchanged (tested exhaustively in
code/tests/test_lazy_dead_end.py); only checker calls are saved.
"""
import ra_core as rc


def reachable(L, U, W, N, contract='referential'):
    R = set(U) - set(W) - set(N)
    cand = set(L) & (set(W) | R)
    if not W:
        return True  # rejection still reachable
    if contract == 'any_valid':
        return bool(cand)
    if len(cand) >= 2:
        return True
    return any(set(W) <= {g} for g in cand)


def lazy_dead_end(L, U, feasible_of, contract='referential', order='lower_first', budget=None):
    L, U = frozenset(L), frozenset(U)
    seq = sorted(U) if order == 'lexical' else sorted(L) + sorted(U - L)
    W, N, queried, work, stopped = set(), set(), [], 0, False
    for g in seq:
        d, sel = rc.decide_sets(L, U, W, N, contract)
        if d != 'ABSTAIN':
            break
        if not reachable(L, U, W, N, contract):
            stopped = True
            break
        if budget is not None and len(queried) >= budget:
            break
        f, w = feasible_of(g)
        queried.append(g)
        work += w
        (W if f else N).add(g)
    d, sel = rc.decide_sets(L, U, W, N, contract)
    return {'decision': d, 'selected': sel, 'queries': len(queried), 'work': work, 'queried': queried,
            'dead_end_stop': stopped}
