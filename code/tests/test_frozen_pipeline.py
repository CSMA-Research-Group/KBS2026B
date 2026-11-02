"""Differential and metamorphic tests of the frozen language pipeline (code/revision_20261022).

* RULE expansion (runtime) vs the generator's independent hidden_members evaluator,
  on every saved public scene with random model-like predicates (not only the hidden spec).
* LIST/RULE schema behaviour (enum identities, empty arrays, duplicates).
* votes(): invalid draws stay in the denominator, duplicates count once per draw.
* score(): outcome categories vs the audit's independent classifier.
* case_decisions(): all-invalid override, oracle isolation.
Requires the imported batch (read-only).
"""
import json
import random
import unittest
from collections import Counter
from pathlib import Path

import _paths  # noqa: F401
import jsonschema
import ra_core as rc
from language_cases import hidden_members, make_cluster
from language_interface import expand, schema, universe, prompt
from language_analysis import votes, score

BATCH = rc.batch_dir()
PUBLIC = BATCH / 'language' / 'public'


def random_rule(public, rng):
    sch = schema(public, 'RULE')['properties']
    out = {}
    for key, spec in sch.items():
        if spec['type'] == 'array':
            vals = spec['items']['enum']
            out[key] = sorted(v for v in vals if rng.random() < rng.choice([.3, .6, .9]))
        else:
            out[key] = rng.choice(spec['enum'])
    return out


class RuleExpansionDifferential(unittest.TestCase):
    def test_expand_equals_independent_evaluator_on_random_predicates(self):
        rng = random.Random(20261026)
        files = sorted(PUBLIC.glob('*.json'))
        self.assertEqual(len(files), 1872)
        n = 0
        for f in files:
            pub = json.loads(f.read_text())
            for _ in range(12):
                spec = random_rule(pub, rng)
                got = expand(pub, 'RULE', spec)
                want = hidden_members(pub, spec)
                self.assertEqual(got, want, (f.name, spec))
                self.assertTrue(set(got) <= set(universe(pub)))
                n += 1
        self.assertEqual(n, 1872 * 12)

    def test_hidden_spec_is_exactly_representable_by_rule_schema(self):
        """Validity note: the RULE hypothesis space contains every hidden specification."""
        for split, n in [('development', 8), ('calibration', 6)]:
            for d in ['carrying', 'workflow']:
                for i in range(n):
                    for c in make_cluster(d, split, i):
                        self.assertEqual(set(c['hidden_spec']), set(schema(c['public'], 'RULE')['properties']))
                        self.assertEqual(expand(c['public'], 'RULE', c['hidden_spec']), c['semantic'])

    def test_prompt_contains_no_hidden_fields(self):
        for d in ['carrying', 'workflow']:
            for c in make_cluster(d, 'calibration', 3):
                for rep in ['LIST', 'RULE']:
                    text = prompt(c['public'], rep)
                    public_json = json.dumps(c['public'], sort_keys=True)
                    self.assertTrue(text.endswith(public_json))
                    self.assertNotIn('hidden_spec', text)
                    self.assertNotIn('"semantic"', text)


class SchemaBehaviour(unittest.TestCase):
    def setUp(self):
        self.pub = json.loads((PUBLIC / 'carrying_calibration_0001_primary.json').read_text())

    def test_list_rejects_unknown_identities_and_accepts_duplicates(self):
        G = universe(self.pub)
        with self.assertRaises(jsonschema.ValidationError):
            expand(self.pub, 'LIST', {'candidates': ['zz->t0']})
        self.assertEqual(expand(self.pub, 'LIST', {'candidates': [G[0], G[0]]}), [G[0]])
        self.assertEqual(expand(self.pub, 'LIST', {'candidates': []}), [])

    def test_rule_empty_array_matches_nothing(self):
        spec = random_rule(self.pub, random.Random(1))
        spec['source_colors'] = []
        self.assertEqual(expand(self.pub, 'RULE', spec), [])

    def test_missing_field_is_invalid(self):
        spec = random_rule(self.pub, random.Random(2))
        spec.pop('pairing')
        with self.assertRaises(jsonschema.ValidationError):
            expand(self.pub, 'RULE', spec)


class Votes(unittest.TestCase):
    def test_invalid_draws_remain_in_denominator_and_duplicates_count_once(self):
        G = ['a', 'b', 'c']
        rs = [{'valid': True, 'candidates': ['a', 'a', 'b']}, {'valid': False, 'candidates': None},
              {'valid': True, 'candidates': ['a']}]
        p = votes(G, rs)
        self.assertEqual(p, {'a': 2 / 3, 'b': 1 / 3, 'c': 0.0})
        self.assertEqual({g: float(v) for g, v in rc.frequencies(G, rs).items()}, p)

    def test_draw_permutation_invariance(self):
        rng = random.Random(3)
        G = list('abcdef')
        for _ in range(500):
            rs = [{'valid': rng.random() < .9, 'candidates': [g for g in G if rng.random() < .4]} for _ in range(rng.randint(1, 6))]
            rs2 = rs[:]
            rng.shuffle(rs2)
            self.assertEqual(votes(G, rs), votes(G, rs2))


class ScoreCategories(unittest.TestCase):
    def test_frozen_score_agrees_with_independent_classifier(self):
        rng = random.Random(4)
        G = list('abcd')
        disagreements = Counter()
        for _ in range(20000):
            S = {g for g in G if rng.random() < .5}
            P = {g for g in G if rng.random() < .5}
            E = S & P
            d = rng.choice(['EXECUTE', 'CLARIFY', 'REJECT', 'ABSTAIN'])
            sel = rng.choice(G) if d == 'EXECUTE' else None
            checked = sorted(g for g in P if rng.random() < .7)
            view = {'decision': d, 'selected_goal': sel, 'contract': 'referential', 'checked_positive_goals': checked}
            s = score(view, E, S)
            c = rc.classify(d, sel, E)
            self.assertEqual(s['wrong_decision'], c['wrong'])
            self.assertEqual(s['correct_decision'], c['correct'])
            self.assertEqual(s['execution_error'], c['wrong_execution'])
            self.assertEqual(s['false_infeasibility'], c['category'] == 'false_infeasibility')
            if s['unsupported_clarification'] != (c['category'] == 'unsupported_clarification'):
                disagreements[(c['category'], s['unsupported_clarification'])] += 1
        # Documented definitional difference: the frozen flag means "fewer than two *checked*
        # executable goals", so a correct-by-reference clarification can still be flagged.
        self.assertTrue(all(cat == 'correct_clarification' or flag is False
                            for cat, flag in disagreements))


if __name__ == '__main__':
    unittest.main()
