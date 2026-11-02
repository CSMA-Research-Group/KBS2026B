"""Tests for the Experiment H scene transform (code/audit/h_generate.py)."""
import sys
import unittest
from pathlib import Path

import _paths  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'audit'))
import h_generate as hg  # noqa: E402
import physics_independent as phy  # noqa: E402
from language_cases import make_cluster, hidden_members  # noqa: E402


class Transform(unittest.TestCase):
    def test_only_selected_distractors_change_and_become_feasible(self):
        changed_total = 0
        for split, i in [('calibration', k) for k in range(12)] + [('test', k) for k in range(12)]:
            base = make_cluster('carrying', split, i)[0]
            a = hg.transform(base, 0.25)
            b = hg.transform(base, 0.25)
            self.assertEqual(a, b)  # deterministic
            ra, rb = a['public']['runtime'], base['public']['runtime']
            self.assertEqual(ra['instruction'], rb['instruction'])
            self.assertEqual(a['hidden_spec'], base['hidden_spec'])
            self.assertEqual({k: ra[k] for k in ['grid', 'robot', 'targets']}, {k: rb[k] for k in ['grid', 'robot', 'targets']})
            before = {o['id']: o for o in rb['objects']}
            reach = set(hg.reachable_cells(rb))
            feas, _ = phy.feasible(a['public'])
            for o in ra['objects']:
                if o['id'] in a['experiment_H']['feasible_distractors']:
                    changed_total += 1
                    self.assertTrue(o['id'].startswith('d') and o['movable'] and o['mass'] <= ra['robot']['payload'])
                    self.assertIn(tuple(o['cell']), reach)
                    self.assertTrue(any(g.startswith(o['id'] + '->') for g in feas))  # really feasible now
                else:
                    self.assertEqual(o, before[o['id']])
            self.assertEqual(a['semantic'], hidden_members(a['public'], a['hidden_spec']))
        self.assertGreater(changed_total, 0)

    def test_pi_zero_is_identity_on_scene(self):
        base = make_cluster('carrying', 'test', 3)[0]
        a = hg.transform(base, 0.0)
        self.assertEqual(a['public'], base['public'])
        self.assertEqual(a['semantic'], base['semantic'])


if __name__ == '__main__':
    unittest.main()
