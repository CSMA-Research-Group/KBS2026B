"""Adversarial boundary worlds for the frozen per-goal checker (runtime_checks.check).

Three-way agreement: frozen per-goal checker, the generator's reference search, and the
audit's independent implementation.  Positive results must carry valid witnesses;
negative results must be complete class-level closures, never budget/timeouts.
"""
import copy
import unittest

import _paths  # noqa: F401
import physics_independent as phy
from runtime_checks import check, decide
from language_cases import physical_reference
from innovation_stage2.domain import validate_witness


DECOY_CELL = (8, 6)


def world(blocked=(), objects=None, targets=None, start=(1, 1), payload=5, w=9, h=7):
    # Decoys guarantee >=2 object and target colours (the reference search requires a
    # disjunction of at least two colours).  Decoy goals are infeasible:
    # the decoy object is immovable and the decoy target is walled in.
    objects = (objects or []) + [obj('zz', (8, 0), movable=False, color='yellow')]
    targets = (targets or []) + [tgt('zt', DECOY_CELL, color='purple')]
    blocked = [tuple(c) for c in blocked if tuple(c) not in {(8, 0), DECOY_CELL}]
    blocked += [c for c in ring(DECOY_CELL) if 0 <= c[0] < w and 0 <= c[1] < h and c not in blocked]
    return {'domain': 'carrying', 'context': {'speaker': 'x', 'owners': {}, 'near_distance': 3},
            'runtime': {'instruction': 'test', 'task_contract': {'mode': 'referential'},
                        'grid': {'width': w, 'height': h, 'blocked': [list(c) for c in blocked]},
                        'robot': {'start': list(start), 'payload': payload},
                        'objects': objects or [], 'targets': targets or []}}


def obj(i, cell, fp=(1, 1), o=0, mass=1, movable=True, color='red'):
    return {'id': i, 'cell': list(cell), 'color': color, 'shape': 'block', 'mass': mass, 'movable': movable,
            'footprint': list(fp), 'orientation': o}


def tgt(i, cell, allowed=(0, 1), color='blue', accepts=('green',)):
    return {'id': i, 'cell': list(cell), 'color': color, 'shape': 'tray', 'allowed_orientations': list(allowed),
            'accepts_colors': list(accepts)}


def ring(c):
    x, y = c
    return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


class ThreeWay(unittest.TestCase):
    def labels(self, pub):
        goals = [s['id'] + '->' + t['id'] for s in pub['runtime']['objects'] for t in pub['runtime']['targets']]
        per_goal = {}
        for g in goals:
            r = check(pub, g)
            per_goal[g] = r['feasible']
            if r['feasible']:
                s, t = g.split('->')
                wit = r['raw']['witness']
                self.assertTrue(validate_witness(pub['runtime'], {'action': 'place', 'source_id': s, 'target_id': t}, wit))
            else:
                self.assertEqual(r['raw']['decision'], 'REJECT')  # complete negative, not a budget stop
        ref = set(physical_reference(pub)['feasible'])
        mine, _ = phy.feasible(pub)
        self.assertEqual({g for g, f in per_goal.items() if f}, ref)
        self.assertEqual(ref, mine)
        return {g: f for g, f in per_goal.items() if not (g.startswith('zz') or g.endswith('zt'))}

    def test_unreachable_target_walled_in(self):
        pub = world(blocked=ring((6, 4)), objects=[obj('o1', (2, 2))], targets=[tgt('t1', (6, 4)), tgt('t2', (3, 4))])
        lab = self.labels(pub)
        self.assertEqual(lab, {'o1->t1': False, 'o1->t2': True})

    def test_immovable_and_overweight_sources(self):
        pub = world(objects=[obj('o1', (2, 2), movable=False), obj('o2', (3, 2), mass=9), obj('o3', (4, 2))],
                    targets=[tgt('t1', (6, 4))])
        self.assertEqual(self.labels(pub), {'o1->t1': False, 'o2->t1': False, 'o3->t1': True})

    def test_competing_feasible_goals_and_colour_is_not_physical(self):
        pub = world(objects=[obj('o1', (2, 2), color='red'), obj('o2', (2, 4), color='green')],
                    targets=[tgt('t1', (6, 2), accepts=('green',)), tgt('t2', (6, 4), accepts=('red',))])
        self.assertTrue(all(self.labels(pub).values()))

    def test_footprint_too_wide_for_corridor(self):
        # One-cell gap at (4,3) between two rooms: a 3x3 object cannot pass in any orientation,
        # a 1x1 object can.  (A 1x3 object CAN pass: it rotates in the open room first.)
        blocked = [(4, y) for y in range(7) if y != 3]
        pub = world(blocked=blocked, objects=[obj('o1', (2, 3), fp=(3, 3)), obj('o2', (2, 1)),
                                              obj('o3', (2, 3), fp=(1, 3))],
                    targets=[tgt('t1', (6, 3))])
        lab = self.labels(pub)
        self.assertEqual(lab['o2->t1'], True)
        self.assertEqual(lab['o1->t1'], False)
        self.assertEqual(lab['o3->t1'], True)

    def test_orientation_requirement_without_rotation_space(self):
        # 3x1 object in a one-row corridor cannot rotate; target only accepts the other orientation.
        blocked = [(x, y) for x in range(9) for y in range(7) if y != 3]
        pub = world(blocked=blocked, start=(1, 3), objects=[obj('o1', (2, 3), fp=(3, 1), o=0)],
                    targets=[tgt('t1', (6, 3), allowed=(1,)), tgt('t2', (5, 3), allowed=(0,))])
        self.assertEqual(self.labels(pub), {'o1->t1': False, 'o1->t2': True})

    def test_multiple_paths_same_goal_class(self):
        pub = world(objects=[obj('o1', (2, 3))], targets=[tgt('t1', (6, 3))])
        alt = copy.deepcopy(pub)
        alt['runtime']['grid']['blocked'] = [[4, 3]]  # forces a detour; class unchanged
        self.assertEqual(self.labels(pub), self.labels(alt))


class LatentReferenceLimitation(unittest.TestCase):
    def test_reference_search_requires_two_object_and_two_target_colours(self):
        """Documents a latent robustness bug in the frozen physical_reference (never triggered by the
        generator, all 1872 saved scenes have >=2 colours): a single-colour scene raises ValueError.
        If this test starts failing, the frozen reference was changed; re-freeze and re-audit."""
        pub = {'domain': 'carrying', 'context': {'speaker': 'x', 'owners': {}, 'near_distance': 3},
               'runtime': {'instruction': 't', 'task_contract': {'mode': 'referential'},
                           'grid': {'width': 5, 'height': 5, 'blocked': []}, 'robot': {'start': [0, 0], 'payload': 5},
                           'objects': [obj('o1', (1, 1))], 'targets': [tgt('t1', (3, 3))]}}
        with self.assertRaises(ValueError):
            physical_reference(pub)
        self.assertEqual(phy.feasible(pub)[0], {'o1->t1'})  # the independent model handles it


class NoNegativeEvidenceFromFailures(unittest.TestCase):
    def test_checker_exception_is_not_converted_to_infeasibility(self):
        def broken(g):
            raise TimeoutError('simulated checker timeout')
        with self.assertRaises(TimeoutError):
            decide({'a'}, {'a', 'b'}, broken)

    def test_budget_truncation_adds_no_negative_evidence(self):
        v = decide(set(), {'a', 'b'}, lambda g: {'feasible': False, 'work_units': 1}, budget=1)
        self.assertEqual(v['decision'], 'ABSTAIN')  # one negative is not a closure of U

    def test_unknown_goal_identity_rejected(self):
        pub = world(objects=[obj('o1', (2, 2))], targets=[tgt('t1', (6, 4))])
        with self.assertRaises(ValueError):
            check(pub, 'o9->t1')


if __name__ == '__main__':
    unittest.main()
