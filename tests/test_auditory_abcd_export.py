"""Synthetic-only checks for one-file P06/P07 A/B/C/D export.

No real participant values or test-outcome metrics are read here.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"prototype"/"auditory"))
from build_abcd_cases import (
    FEATURES, IDS, STAGE_TEXT, ExportError, frozen_regression_coefficients,
    infer_fuzzy_rules, infer_rule_strength, load_legacy, load_saved_partitions, numeric,
)


def synthetic_legacy():
    measurements=[
        {"Name":"Mental demand","Observed":5,"ShapContribution":0.4,"Unit":"score /10"},
        {"Name":"Time between qualifying grabs","Observed":4,"ShapContribution":-0.1,"Unit":"seconds"},
        {"Name":"Time to locate target","Observed":3,"ShapContribution":0.2,"Unit":"seconds"},
        {"Name":"Head turning","Observed":40,"ShapContribution":0.1,"Unit":"degrees"},
    ]
    return [
        {
            "Case":"A" if person=="P06" else "B",
            "ParticipantId":person,"DifficultyStage":stage,
            "Regression":{"Measurements":measurements,"ReferencePrediction":1.2,"Prediction":1.8},
            "Fuzzy":{"Recommendations":[{},{}],"Rules":[{} for _ in range(10)]}
        }
        for person in IDS for stage in STAGE_TEXT.values()
    ]


class UnifiedExporterTests(unittest.TestCase):
    def test_accept_previously_inspected_p06_p07_source_in_any_row_order(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"ab.json"
            p.write_text(json.dumps(synthetic_legacy()[::-1]),encoding="utf-8")
            self.assertEqual(len(load_legacy(p)),8)

    def test_accept_dutch_decimal_comma_not_thousands_separator(self):
        self.assertAlmostEqual(numeric("4,39999999999997", "measurement"),
                               4.39999999999997)
        with self.assertRaisesRegex(ExportError, "Invalid original A/B number"):
            numeric("4.123,4", "measurement")

    def test_legacy_json_decimal_comma_roundtrips(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "ab.json"
            rows = synthetic_legacy()
            rows[0]["Regression"]["Measurements"][0]["Observed"] = "5,0"
            rows[0]["Regression"]["Measurements"][0]["ShapContribution"] = "0,4"
            rows[0]["Regression"]["Prediction"] = "1,8"
            p.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(len(load_legacy(p)), 8)

    def test_reject_substituted_test_person(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"ab.json"
            data=synthetic_legacy()
            data[0]["ParticipantId"]="P08"
            p.write_text(json.dumps(data),encoding="utf-8")
            with self.assertRaisesRegex(ExportError,"P06 and P07"):
                load_legacy(p)

    def test_reject_legacy_shap_breakdown_not_matching_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"ab.json"
            data=synthetic_legacy()
            data[0]["Regression"]["Prediction"]=2.1
            p.write_text(json.dumps(data),encoding="utf-8")
            with self.assertRaisesRegex(ExportError,"SHAP"):
                load_legacy(p)

    def test_frozen_regression_coefficients_must_have_four_features_and_intercept(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"coeff.csv"
            rows=[{"model":"auditory","predictor":"Intercept","coefficient":-0.4}]
            rows.extend({"model":"auditory","predictor":v,"coefficient":0.2} for v in FEATURES)
            pd.DataFrame(rows).to_csv(path,index=False)
            self.assertEqual(set(frozen_regression_coefficients(path)),
                             {"Intercept",*FEATURES})

    def test_interpolate_memberships_using_original_piecewise_linear_knots(self):
        f=FEATURES[0]
        partitions={f:(["Low","High"],np.array([0.,5.,10.]),
                       np.array([[1.,0.],[.5,.5],[0.,1.]]))}
        self.assertAlmostEqual(
            infer_rule_strength([[f,["High"]]],{f:7.5},partitions),0.75)
        self.assertAlmostEqual(
            infer_rule_strength([[f,["Low","High"]]],{f:7.5},partitions),0.75)
        self.assertAlmostEqual(
            infer_rule_strength([[f,["Low"]]],{f:-2.},partitions),1.0)

    def test_load_all_four_saved_input_partitions_from_original_knots(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"partitions.csv"
            knots=json.dumps([[0.,[1.,0.]],[5.,[.5,.5]],[10.,[0.,1.]]])
            rows=[]
            for feature in FEATURES:
                for i,name in enumerate(["Low","High"]):
                    rows.append({"feature":feature,"term":name,
                                 "region_index":i,"knots_json":knots})
            pd.DataFrame(rows).to_csv(p,index=False)
            result=load_saved_partitions(p)
            self.assertEqual(set(result),set(FEATURES))
            self.assertAlmostEqual(
                infer_rule_strength([[FEATURES[3],["High"]]],
                                    {FEATURES[3]:5.},result),0.5)

    def test_five_term_mamdani_inference_reconstructs_numeric_centroid(self):
        f=FEATURES[0]
        rules=pd.DataFrame([
            {"source_leaf_ids":json.dumps([i]),
             "antecedent_or_groups_json":json.dumps([[f,["Low"]]])}
            for i in range(8)
        ])
        mapping=pd.DataFrame([
            {"configuration":"output_5_terms","source_leaf_ids":json.dumps([i]),
             "output_consequent":"Medium"} for i in range(8)
        ])
        partitions={f:(["Low","High"],np.array([0.,10.]),
                       np.array([[1.,0.],[0.,1.]]))}
        rows,pred,grid,curve=infer_fuzzy_rules(
            partitions,rules,mapping,{f:0.},np.linspace(0,4,5))
        self.assertEqual(len(rows),8)
        self.assertEqual(len(grid),4001)
        self.assertTrue(all(r["activation"]==1. for r in rows))
        self.assertAlmostEqual(pred,2.,places=10)
        self.assertTrue(np.isfinite(curve).all())

    def test_no_active_rule_preserves_unavailable(self):
        f=FEATURES[0]
        rules=pd.DataFrame([
            {"source_leaf_ids":json.dumps([i]),
             "antecedent_or_groups_json":json.dumps([[f,["High"]]])}
            for i in range(8)
        ])
        mapping=pd.DataFrame([
            {"configuration":"output_5_terms","source_leaf_ids":json.dumps([i]),
             "output_consequent":"Medium"} for i in range(8)
        ])
        partitions={f:(["Low","High"],np.array([0.,10.]),
                       np.array([[1.,0.],[0.,1.]]))}
        _,pred,_,_ =infer_fuzzy_rules(
            partitions,rules,mapping,{f:0.},np.linspace(0,4,5))
        self.assertIsNone(pred)


if __name__=="__main__":
    unittest.main()
