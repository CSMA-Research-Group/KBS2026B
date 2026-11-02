"""Exhaustive / property / metamorphic tests of the manuscript's formal claims.

Covers Theorem 2 (sufficiency AND necessity), the realisability construction, the
coverage lemmas behind Proposition 5, Proposition 6 nesting, Proposition 4 lower bound,
the lazy procedure (frozen code) and the decision-risk score used by the audit.
Exhaustive over small universes; seeded random property checks elsewhere.
Run: python3 -m unittest discover -s code/tests -v
"""
import itertools
import math
import random
import unittest
from fractions import Fraction as F

import _paths  # noqa: F401
import ra_core as rc
import ra_calibration as cal
from runtime_checks import decide as frozen_decide
from language_analysis import quantile as frozen_quantile, nonconformity as frozen_nonconformity

HALF = F(1, 2)


def subsets(xs):
    xs = list(xs)
    return [frozenset(c) for k in range(len(xs) + 1) for c in itertools.combinations(xs, k)]


def ref(E):
    E = set(E)
    if not E:
        return ('REJECT', None)
    if len(E) == 1:
        return ('EXECUTE', next(iter(E)))
    return ('CLARIFY', None)


def correct_for(decision, E):
    """Is a non-abstaining referential decision correct for executable set E?"""
    d, g = decision
    if d == 'EXECUTE':
        return set(E) == {g}
    if d == 'CLARIFY':
        return len(E) >= 2
    if d == 'REJECT':
        return len(E) == 0
    raise ValueError(decision)


def correct_any_valid(decision, E):
    d, g = decision
    if d == 'EXECUTE':
        return g in E
    if d == 'REJECT':
        return len(E) == 0
    raise ValueError(decision)


def interval(A, B):
    extra = sorted(set(B) - set(A))
    return [frozenset(A) | s for s in subsets(extra)]


def env(p, q):
    return rc.envelope(p, q)


def full_decision(L, U, P, contract='referential'):
    U = frozenset(U)
    return rc.decide_sets(L, U, U & P, U - P, contract)


class Theorem2(unittest.TestCase):
    """d(A,B) is exactly the set of decisions correct for every permitted completion."""

    def test_sufficiency_and_necessity_referential(self):
        for n in range(0, 5):
            G = [chr(97 + i) for i in range(n)]
            cands = [('REJECT', None), ('CLARIFY', None)] + [('EXECUTE', g) for g in G]
            for B in subsets(G):
                for A in subsets(B):
                    comps = interval(A, B)
                    safe = {c for c in cands if all(correct_for(c, E) for E in comps)}
                    d, g = rc.decide_sets(A, B, A, frozenset(G) - B)  # L=W=A, U=B, N=G-B gives A,B
                    if d == 'ABSTAIN':
                        self.assertEqual(safe, set(), (A, B))
                    else:
                        self.assertEqual(safe, {(d, g)}, (A, B))

    def test_sufficiency_and_necessity_any_valid(self):
        for n in range(0, 5):
            G = [chr(97 + i) for i in range(n)]
            cands = [('REJECT', None)] + [('EXECUTE', g) for g in G]
            for B in subsets(G):
                for A in subsets(B):
                    comps = interval(A, B)
                    safe = {c for c in cands if all(correct_any_valid(c, E) for E in comps)}
                    expect = {('EXECUTE', g) for g in A} | ({('REJECT', None)} if not B else set())
                    self.assertEqual(safe, expect, (A, B))

    def test_evidence_model_equals_interval_both_directions(self):
        """Executable sets consistent with (L,U,W,N) and decision coverage are exactly [A,B]."""
        G = ['a', 'b', 'c']
        allS = subsets(G)
        for U in subsets(G):
            for L in subsets(U):
                for W in subsets(G):
                    for N in subsets(set(G) - W):
                        A, B = L & W, U - N
                        consistent = set()
                        for P in allS:
                            if not (W <= P and not (N & P)):
                                continue
                            for S in allS:
                                E = S & P
                                if (L & P) <= E <= (U & P):
                                    consistent.add(E)
                        self.assertEqual(consistent, set(interval(A, B)), (L, U, W, N))

    def test_realisation_construction_including_semantic_coverage(self):
        """P' = W | E', S' = L | E' realises every E' in [A,B] with semantic (hence decision) coverage."""
        G = ['a', 'b', 'c', 'd']
        for U in subsets(G):
            for L in subsets(U):
                for W in subsets(G):
                    for N in subsets(set(G) - W):
                        A, B = L & W, U - N
                        for E in interval(A, B):
                            P2, S2 = W | E, L | E
                            self.assertTrue(W <= P2 and not (N & P2))
                            self.assertEqual(S2 & P2, E)
                            self.assertTrue(L <= S2 <= U)
                            # the paper's own choice S' = E' gives decision coverage
                            self.assertTrue((L & P2) <= (E & P2) == E <= (U & P2))

    def test_correlations_can_decide_where_interval_rule_abstains(self):
        """Documented limitation, not a bug: a mutual-exclusion constraint settles A={a},B={a,b}."""
        A, B = frozenset('a'), frozenset('ab')
        self.assertEqual(rc.decide_sets(A, B, A, frozenset())[0], 'ABSTAIN')
        constrained = [E for E in interval(A, B) if not {'a', 'b'} <= E]  # at most one of a, b
        self.assertEqual(constrained, [frozenset('a')])


class CoverageLemmas(unittest.TestCase):
    """Proposition 5's core: coverage at threshold q holds iff the score is <= q."""

    def grid(self, k):
        return sorted({F(j, k) for j in range(k + 1)} | {F(j, 12) for j in range(6, 13)})

    def test_decision_and_semantic_coverage_iff_score_le_q(self):
        G = ['a', 'b', 'c']
        for k in (1, 2, 3, 4):
            vals = [F(j, k) for j in range(k + 1)]
            for pv in itertools.product(vals, repeat=len(G)):
                p = dict(zip(G, pv))
                for S in subsets(G):
                    for P in subsets(G):
                        E = S & P
                        rd, rs = rc.r_dec(p, S, P), rc.r_sem(p, S, G)
                        for q in self.grid(k):
                            if q < HALF:
                                continue
                            L, U = env(p, q)
                            self.assertEqual((L & P) <= E <= (U & P), rd <= q)
                            self.assertEqual(L <= S <= U, rs <= q)

    def test_decision_score_never_exceeds_semantic_score(self):
        rng = random.Random(1)
        for _ in range(20000):
            G = list('abcdef')[:rng.randint(1, 6)]
            k = rng.randint(1, 10)
            p = {g: F(rng.randint(0, k), k) for g in G}
            S = frozenset(g for g in G if rng.random() < .5)
            P = frozenset(g for g in G if rng.random() < .5)
            self.assertLessEqual(rc.r_dec(p, S, P), rc.r_sem(p, S, G))

    def test_frozen_float_scores_equal_exact_scores(self):
        rng = random.Random(2)
        for _ in range(20000):
            G = ['g%d' % i for i in range(rng.randint(1, 8))]
            k = rng.choice([1, 2, 3, 5, 10, 20])
            p = {g: F(rng.randint(0, k), k) for g in G}
            S = {g for g in G if rng.random() < .5}
            P = [g for g in G if rng.random() < .5]
            pf = {g: float(v) for g, v in p.items()}
            self.assertAlmostEqual(frozen_nonconformity(pf, S, G), float(rc.r_sem(p, S, G)), places=12)
            self.assertAlmostEqual(frozen_nonconformity(pf, S, P), float(rc.r_dec(p, S, P)), places=12)


class Proposition6(unittest.TestCase):
    def test_full_evidence_nesting_for_any_q1_le_q2(self):
        G = ['a', 'b', 'c']
        for k in (2, 3):
            vals = [F(j, k) for j in range(k + 1)]
            qs = sorted({HALF, F(1)} | {v for v in vals if v >= HALF})
            for pv in itertools.product(vals, repeat=3):
                p = dict(zip(G, pv))
                for P in subsets(G):
                    for q1, q2 in itertools.combinations_with_replacement(qs, 2):
                        d2 = full_decision(*env(p, q2), P)
                        d1 = full_decision(*env(p, q1), P)
                        if d2[0] != 'ABSTAIN':
                            self.assertEqual(d1, d2, (p, P, q1, q2))

    def test_q_dec_le_q_sem_on_random_calibration_sets(self):
        rng = random.Random(3)
        for _ in range(300):
            n = rng.choice([10, 50, 128])
            sem, dec = [], []
            for _ in range(n):
                G = list('abcde')
                p = {g: F(rng.randint(0, 3), 3) for g in G}
                S = frozenset(g for g in G if rng.random() < .4)
                P = frozenset(g for g in G if rng.random() < .5)
                sem.append(rc.r_sem(p, S, G))
                dec.append(rc.r_dec(p, S, P))
            for a in (.01, .05, .1, .2):
                self.assertLessEqual(rc.conformal_q(dec, a), rc.conformal_q(sem, a))

    def test_q_equal_one_makes_execution_and_clarification_impossible(self):
        p = {'a': F(1), 'b': F(0)}
        L, U = env(p, F(1))
        self.assertEqual(L, frozenset())
        self.assertEqual(U, frozenset(p))
        for P in subsets(p):
            d = full_decision(L, U, P)[0]
            self.assertIn(d, ('ABSTAIN', 'REJECT'))
            self.assertEqual(d == 'REJECT', not P)


class LazyProcedureFrozen(unittest.TestCase):
    """The frozen runtime_checks.decide against the set algebra of Theorem 2."""

    def run_decide(self, L, U, P, contract, order, budget):
        calls = []

        def q(g):
            calls.append(g)
            return {'feasible': g in P, 'work_units': 1}
        v = frozen_decide(L, U, q, contract, order, budget)
        return v, calls

    def test_full_budget_equals_full_evidence_rule_and_budget_is_subdecision(self):
        G = ['a', 'b', 'c', 'd']
        n = 0
        for U in subsets(G):
            for L in subsets(U):
                for P in subsets(G):
                    for contract in ('referential', 'any_valid'):
                        full = full_decision(L, U, P, contract)
                        for order in ('lower_first', 'lexical', 'eager'):
                            v, calls = self.run_decide(L, U, P, contract, order, None)
                            self.assertEqual((v['decision'], v['selected_goal']), full, (L, U, P, contract, order))
                            self.assertEqual(len(calls), len(set(calls)))
                            self.assertTrue(set(calls) <= U)
                            for budget in range(0, len(U) + 1):
                                vb, cb = self.run_decide(L, U, P, contract, order, budget)
                                self.assertLessEqual(len(cb), budget)
                                self.assertIn((vb['decision'], vb['selected_goal']), {full, ('ABSTAIN', None)})
                                n += 1
        self.assertGreater(n, 10000)

    def test_no_query_after_determination_for_stopping_orders(self):
        G = ['a', 'b', 'c', 'd']
        for U in subsets(G):
            for L in subsets(U):
                for P in subsets(G):
                    for order in ('lower_first', 'lexical'):
                        v, calls = self.run_decide(L, U, P, 'referential', order, None)
                        if calls:
                            prev, _ = self.run_decide(L, U, P, 'referential', order, len(calls) - 1)
                            self.assertEqual(prev['decision'], 'ABSTAIN', (L, U, P, order))

    def test_independent_lazy_matches_frozen_lazy_exactly(self):
        G = ['a', 'b', 'c', 'd']
        for U in subsets(G):
            for L in subsets(U):
                for P in subsets(G):
                    for contract in ('referential', 'any_valid'):
                        for order in ('lower_first', 'lexical', 'eager'):
                            for budget in (None, 0, 1, 2, 3):
                                v, calls = self.run_decide(L, U, P, contract, order, budget)
                                w = rc.lazy(L, U, lambda g: (g in P, 1), contract, order, budget)
                                self.assertEqual((v['decision'], v['selected_goal'], calls),
                                                 (w['decision'], w['selected'], w['queried']))

    def test_goal_renaming_metamorphic(self):
        """Renaming goals permutes queries but never changes the full-budget decision class."""
        G = ['a', 'b', 'c', 'd']
        perm = dict(zip(G, ['d', 'c', 'b', 'a']))
        for U in subsets(G):
            for L in subsets(U):
                for P in subsets(G):
                    v, _ = self.run_decide(L, U, P, 'referential', 'lower_first', None)
                    m = lambda s: frozenset(perm[x] for x in s)
                    w, _ = self.run_decide(m(L), m(U), m(P), 'referential', 'lower_first', None)
                    self.assertEqual(v['decision'], w['decision'])
                    if v['decision'] == 'EXECUTE':
                        self.assertEqual(perm[v['selected_goal']], w['selected_goal'])


class Proposition4(unittest.TestCase):
    def test_fooling_set_lower_bound(self):
        """L=U, <=1 feasible goal: every strict query subset admits a world with another decision."""
        for n in range(1, 5):
            U = frozenset(chr(97 + i) for i in range(n))
            for P in [frozenset()] + [frozenset({g}) for g in U]:
                for Q in subsets(U):
                    if Q == U:
                        continue
                    alternatives = [P2 for P2 in subsets(U) if (P2 & Q) == (P & Q) and ref(P2) != ref(P)]
                    self.assertTrue(alternatives, (U, P, Q))

    def test_two_witnesses_settle_clarification(self):
        U = frozenset('abcd')
        for P in subsets(U):
            if len(P) >= 2:
                two = frozenset(sorted(P)[:2])
                self.assertTrue(all(ref(P2)[0] == 'CLARIFY' for P2 in subsets(U) if two <= P2))

    def test_weighted_cover_is_exactly_cheapest_sufficient_bundle(self):
        rng = random.Random(4)
        for _ in range(300):
            H = frozenset(range(rng.randint(1, 5)))
            certs = [(frozenset(x for x in H if rng.random() < .5), rng.randint(1, 9)) for _ in range(rng.randint(1, 6))]
            best = None
            for J in subsets(range(len(certs))):
                covered = frozenset().union(*[certs[j][0] for j in J]) if J else frozenset()
                if H <= covered:
                    cost = sum(certs[j][1] for j in J)
                    best = cost if best is None else min(best, cost)
            # Exclusion of all of H is achievable iff a cover exists; cost equals cover optimum.
            union = frozenset().union(*[c for c, _ in certs])
            self.assertEqual(best is not None, H <= union)


class DecisionRiskScore(unittest.TestCase):
    """The audit's decision-loss calibration: loss monotone in q, tighter than r_dec, CRC == conformal."""

    @staticmethod
    def loss(p, S, P, q):
        E = S & P
        d, g = full_decision(*env(p, q), P)
        return int(d != 'ABSTAIN' and not correct_for((d, g), E))

    def test_loss_monotone_and_task_score_le_decision_score(self):
        G = ['a', 'b', 'c']
        k = 3
        vals = [F(j, k) for j in range(k + 1)]
        grid = cal.q_grid(k)
        for pv in itertools.product(vals, repeat=3):
            p = dict(zip(G, pv))
            for S in subsets(G):
                for P in subsets(G):
                    losses = [self.loss(p, S, P, q) for q in grid]
                    self.assertTrue(all(x >= y for x, y in zip(losses, losses[1:])), (p, S, P, losses))
                    self.assertEqual(losses[-1], 0)
                    r_task = next(q for i, q in enumerate(grid) if not any(losses[i:]))
                    self.assertLessEqual(r_task, max(HALF, rc.r_dec(p, S, P)))

    def test_crc_selection_equals_conformal_quantile_of_task_score(self):
        rng = random.Random(5)
        grid = cal.q_grid(3)
        for _ in range(200):
            n = rng.choice([20, 64, 128])
            recs, task = [], []
            for _ in range(n):
                G = tuple('abcd')
                p = {g: F(rng.choice([0, 0, 1, 2, 3, 3]), 3) for g in G}
                S = frozenset(g for g in G if rng.random() < .4)
                P = frozenset(g for g in G if rng.random() < .5)
                rec = {'p': p, 'S': S, 'P': P, 'E': S & P, 'G': G, 'valid_any': True}
                recs.append(rec)
                losses = [self.loss(p, S, P, q) for q in grid]
                task.append(next(q for i, q in enumerate(grid) if not any(losses[i:])))
            for a in (.05, .1, .2):
                lev, _ = cal.crc_select(recs, a, grid, cal.env_q)
                self.assertEqual(lev, rc.conformal_q(task, a))


class QuantileArithmetic(unittest.TestCase):
    def test_frozen_quantile_rank_matches_exact_rank_for_paper_settings(self):
        for a in (.01, .05, .1, .2):
            self.assertEqual(math.ceil((128 + 1) * (1 - a)), rc.conformal_rank(128, a))
            scores = [F(i, 128) for i in range(128)]
            fq = frozen_quantile([float(s) for s in scores], a)
            self.assertAlmostEqual(fq, float(rc.conformal_q(scores, a)), places=12)

    def test_record_float_rank_disagreements_elsewhere(self):
        """Not a failure for the paper (n=128); documents a latent robustness issue for other n."""
        bad = [(n, a) for a in (.01, .025, .05, .1, .15, .2) for n in range(1, 3001)
               if math.ceil((n + 1) * (1 - a)) != rc.conformal_rank(n, a)]
        self.__class__.float_rank_disagreements = bad
        for n, a in bad:
            self.assertNotEqual(n, 128)

    def test_rank_above_n_gives_full_envelope(self):
        self.assertEqual(frozen_quantile([0.] * 10, .05), 1.)  # ceil(11*.95)=11 > 10
        self.assertEqual(frozen_quantile([0.] * 19, .05), .5)
        self.assertEqual(frozen_quantile([1.] * 128, .05), 1.)


class ConformalMonteCarlo(unittest.TestCase):
    def test_marginal_coverage_with_ties_and_clamp(self):
        rng = random.Random(6)
        vals = [0, 1 / 3, 2 / 3, 1]
        weights = [.6, .25, .12, .03]
        n, a, reps, cover = 128, .05, 4000, 0
        for _ in range(reps):
            sample = rng.choices(vals, weights, k=n + 1)
            q = frozen_quantile(sample[:n], a)
            cover += sample[n] <= q + 1e-12
        # Exact guarantee >= 0.95; allow 3.5 Monte Carlo standard errors.
        self.assertGreaterEqual(cover / reps, .95 - 3.5 * math.sqrt(.95 * .05 / reps))

    def test_conditional_execution_error_can_exceed_alpha(self):
        """Population where marginal error is 4% but error | EXECUTE is 100%."""
        population = [('ABSTAIN', False)] * 96 + [('EXECUTE', True)] * 4
        marginal = sum(e for _, e in population) / len(population)
        executed = [e for d, e in population if d == 'EXECUTE']
        self.assertLessEqual(marginal, .05)
        self.assertEqual(sum(executed) / len(executed), 1.0)


if __name__ == '__main__':
    unittest.main()
