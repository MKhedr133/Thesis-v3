"""Synthetic tests for the local Auditory C/D development viewer exporter.

No participant data and no reserved-test samples are used.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prototype" / "auditory"))
from build_cd_examples import ExportError, fuzzy_output, make_tree, membership_matrix, tree_leaf


def eight_leaves():
    paths = []
    def traverse(path, depth, leaf):
        if depth == 3:
            paths.append({"leaf_id": leaf, "tree_output": float(leaf)/4,
                          "conditions_json": json.dumps(path)})
            return
        name = ("mental_demand_score_0_to_10",
                "median_time_to_target_seconds",
                "median_head_turning_degrees")[depth]
        for direction, offset in (("<=", 0), (">", 1)):
            traverse(path + [{"feature": name, "operator": direction,
                              "threshold": 4.0}],depth+1,leaf*2+offset)
    traverse([],0,0)
    return pd.DataFrame(paths)


class DevelopmentExportSyntheticTests(unittest.TestCase):
    def test_tree_paths_recover_eight_unique_leaves(self):
        tree=make_tree(eight_leaves())
        leaves=set()
        for mental in (3.,7.):
            for time in (3.,7.):
                for head in (3.,7.):
                    leaf=tree_leaf(tree,np.array([mental,0.,time,head]))
                    leaves.add(leaf["leaf"])
        self.assertEqual(len(leaves),8)

    def test_reject_incomplete_tree_paths(self):
        with self.assertRaisesRegex(ExportError,"eight"):
            make_tree(eight_leaves().iloc[:-1])

    def test_memberships_sum_to_one_and_match_exact_centres(self):
        centres=np.array([[0.,0.,0.,0.],[1.,0.,0.,0.],[2.,0.,0.,0.]])
        values=np.array([[0.,0.,0.,0.],[1.5,0.,0.,0.]])
        u=membership_matrix(values,centres)
        np.testing.assert_allclose(u.sum(axis=1),1.,atol=1e-12)
        np.testing.assert_allclose(u[0], [1.,0.,0.],atol=1e-12)
        self.assertTrue((u[1]>0).all())

    def test_mamdani_uses_max_of_clipped_rule_consequents(self):
        centres=np.linspace(0.,4.,5)
        rules=[{"consequent":"Low","activation":0.6},
               {"consequent":"High","activation":0.4}]
        prediction,x,output=fuzzy_output(rules,centres)
        self.assertEqual(x.shape,(4001,))
        self.assertEqual(output.shape,(4001,))
        self.assertTrue(np.isfinite(prediction))
        self.assertLessEqual(output.max(),0.6+1e-12)
        self.assertGreater(output.max(),0.5)

    def test_no_active_mamdani_rule_returns_unavailable_not_fallback(self):
        prediction,_,output=fuzzy_output(
            [{"consequent":"Medium","activation":0.}],np.linspace(0.,4.,5))
        self.assertIsNone(prediction)
        self.assertTrue(np.all(output==0))


if __name__=="__main__":
    unittest.main()
