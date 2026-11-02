"""Exhaustive tests for the dead-end (certain-abstention) stopping proposal (ra_lazy)."""
import itertools
import unittest

import _paths  # noqa: F401
import ra_core as rc
import ra_lazy as rl


def subsets(xs):
    xs = list(xs)
    return [frozenset(c) for k in range(len(xs) + 1) for c in itertools.combinations(xs, k)]


class DeadEnd(unittest.TestCase):
    def test_reachability_is_exact(self):
        G = 'abcd'
        for U in subsets(G):
            for L in subsets(U):
                for W in subsets(U):
                    for N in subsets(U - W):
                        R = U - W - N
                        for contract in ('referential', 'any_valid'):
                            outcomes = {rc.decide_sets(L, U, W | X, N | (R - X), contract)[0] for X in subsets(R)}
                            self.assertEqual(rl.reachable(L, U, W, N, contract), outcomes != {'ABSTAIN'},
                                             (L, U, W, N, contract))

    def test_same_decisions_never_more_queries(self):
        G = 'abcd'
        saved = 0
        for U in subsets(G):
            for L in subsets(U):
                for P in subsets(G):
                    for contract in ('referential', 'any_valid'):
                        for order in ('lower_first', 'lexical'):
                            f = lambda g: (g in P, 1)
                            a = rc.lazy(L, U, f, contract, order, None)
                            b = rl.lazy_dead_end(L, U, f, contract, order, None)
                            self.assertEqual((a['decision'], a['selected']), (b['decision'], b['selected']))
                            self.assertLessEqual(b['queries'], a['queries'])
                            saved += a['queries'] - b['queries']
                            for budget in range(len(U) + 1):
                                c = rl.lazy_dead_end(L, U, f, contract, order, budget)
                                self.assertIn((c['decision'], c['selected']), {(a['decision'], a['selected']), ('ABSTAIN', None)})
        self.assertGreater(saved, 0)


if __name__ == '__main__':
    unittest.main()
