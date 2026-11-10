#!/usr/bin/env python3
"""Create ONE local A/B/C/D Auditory viewer JSON for P06 and P07.

Exploratory demonstration only: these reserved-test participants were already
inspected while designing the earlier A/B interface. The frozen development
CART/Mamdani/FCM/Sugeno parameters are reused as-is; no test outcomes are
used to fit, tune, compare or select models. Keep the result Git-ignored.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import MOD_14_direct_linear_regression_evaluation as MOD14
from build_cd_examples import (
    FEATURES, KEYS, LEVELS, VOCAB, ExportError, assemble, fuzzy_output,
    membership_matrix, require, tree_leaf,
)


IDS = ("P06", "P07")
NAME_BY_FEATURE = {
    "mental_demand_score_0_to_10": "Mental demand",
    "median_time_between_qualifying_grabs_seconds": "Time between qualifying grabs",
    "median_time_to_target_seconds": "Time to locate target",
    "median_head_turning_degrees": "Head turning",
}
STAGE_TEXT = {
    "0": "Difficulty Stage 0 (D0)",
    "2": "Difficulty Stage 1 (D2)",
    "6": "Difficulty Stage 2 (D6)",
    "10": "Difficulty Stage 3 (D10)",
}


def numeric(value: object, label: str) -> float:
    """Match the legacy JS parser: comma OR dot decimal, never thousands."""
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str) and re.fullmatch(
        r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:e[+-]?\d+)?",
        value.strip(), flags=re.IGNORECASE,
    ):
        result = float(value.strip().replace(",", "."))
    else:
        raise ExportError("Invalid original A/B number: " + label)
    require(np.isfinite(result), "Nonfinite original A/B number: " + label)
    return result


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy(path: Path) -> list[dict]:
    require(path.is_file(), "Missing previous A/B viewer JSON")
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(records, list) and len(records) == 8,
            "Previous A/B viewer must contain exactly eight observations")
    identities = set()
    seen_cases = {}
    for record in records:
        require(isinstance(record, dict) and record.get("Case") in ("A", "B"),
                "Unexpected previous A/B case")
        participant = record.get("ParticipantId")
        require(participant in IDS, "Expected original A/B labels P06 and P07; do not infer a mapping")
        label = record.get("DifficultyStage")
        require(label in STAGE_TEXT.values(), "Unexpected original stage encoding")
        stage = next(k for k,v in STAGE_TEXT.items() if v == label)
        require((participant,stage) not in identities, "Duplicate P06/P07 stage")
        identities.add((participant,stage))
        if record["Case"] in seen_cases:
            require(seen_cases[record["Case"]] == participant,
                    "One A/B case contains multiple participant IDs")
        seen_cases[record["Case"]] = participant
        regression = record.get("Regression")
        require(isinstance(regression,dict) and
                isinstance(regression.get("Measurements"),list) and
                len(regression["Measurements"])==4,
                "Missing original regression inputs")
        names = [v.get("Name") for v in regression["Measurements"]]
        require(set(names) == set(NAME_BY_FEATURE.values()) and len(set(names)) == 4,
                "Unexpected regression input names")
        observed = {item["Name"]: numeric(item["Observed"], "measured value")
                    for item in regression["Measurements"]}
        require(all(np.isfinite(v) for v in observed.values()),
                "Nonfinite original measured value")
        contrib = [numeric(item["ShapContribution"], "SHAP contribution") for item in regression["Measurements"]]
        reference = numeric(regression["ReferencePrediction"], "stage reference")
        predicted = numeric(regression["Prediction"], "original prediction")
        require(all(np.isfinite(v) for v in [*contrib,reference,predicted]) and
                np.isclose(reference+sum(contrib),predicted,atol=1e-8,rtol=0),
                "Original SHAP attributions do not reconstruct")
        fuzzy = record.get("Fuzzy")
        require(isinstance(fuzzy,dict) and len(fuzzy.get("Recommendations",[])) == 2 and
                len(fuzzy.get("Rules",[])) == 10,
                "Original manual fuzzy rules missing")
    require(set(seen_cases.values())==set(IDS) and
            identities=={(p,s) for p in IDS for s in STAGE_TEXT},
            "Original A/B file must contain P06 and P07 at all four stages")
    return records


def frozen_regression_coefficients(path: Path) -> dict[str,float]:
    require(path.is_file(),"Missing original MOD-14 coefficient file")
    table=pd.read_csv(path)
    require({"model","predictor","coefficient"}.issubset(table),
            "Original coefficient table has changed columns")
    auditory=table.loc[table["model"]=="auditory"]
    expected={"Intercept", *FEATURES}
    require(len(auditory)==5 and set(auditory["predictor"])==expected,
            "Unexpected original Auditory MLR coefficient set")
    fitted=dict(zip(auditory["predictor"],auditory["coefficient"].astype(float)))
    require(all(np.isfinite(v) for v in fitted.values()),"Nonfinite original coefficients")
    return fitted


def load_saved_partitions(path: Path) -> dict:
    require(path.is_file(),"Missing saved tree-aligned input membership partitions")
    table=pd.read_csv(path)
    require({"feature","term","region_index","knots_json"}.issubset(table),
            "Saved membership-partition columns are incomplete")
    partitions={}
    for feature in FEATURES:
        group=table.loc[table["feature"]==feature].sort_values("region_index")
        require(not group.empty,"Missing feature partition: "+feature)
        labels=group["term"].tolist()
        require(len(labels)==len(set(labels)) and group["region_index"].tolist()==list(range(len(group))),
                "Partition term ordering is invalid")
        knots=json.loads(group.iloc[0]["knots_json"])
        for saved in group["knots_json"]:
            require(json.loads(saved)==knots,"Inconsistent knots in one feature partition")
        xx=np.asarray([k[0] for k in knots],dtype=float)
        yy=np.asarray([k[1] for k in knots],dtype=float)
        require(yy.shape==(len(xx),len(labels)) and np.all(np.diff(xx)>0) and
                np.isfinite(xx).all() and np.isfinite(yy).all() and
                np.all(yy>=0) and np.all(yy<=1),
                "Invalid learned membership knots")
        partitions[feature]=(labels,xx,yy)
    return partitions


def infer_rule_strength(antecedent: list, values: dict[str,float], partitions: dict) -> float:
    active=[]
    for feature, options in antecedent:
        require(feature in partitions and isinstance(options,list) and len(options)>0,
                "Unrecognised rule feature/alternative terms")
        labels,knots,weights=partitions[feature]
        matched=[]
        for term in options:
            require(term in labels,"Unknown frozen membership term")
            degree=float(np.interp(values[feature],knots,weights[:,labels.index(term)]))
            matched.append(degree)
        active.append(max(matched))
    return min(active) if active else 1.0


def infer_fuzzy_rules(partitions: dict, frozen_rules: pd.DataFrame,
                      frozen_mapping: pd.DataFrame, values: dict[str,float],
                      centres: np.ndarray) -> tuple[list[dict],float|None,np.ndarray,np.ndarray]:
    chosen=frozen_mapping.loc[frozen_mapping["configuration"]=="output_5_terms"]
    require(len(chosen)==8,"Expected eight original five-output-term consequents")
    output_terms={}
    for row in chosen.itertuples(index=False):
        key=tuple(json.loads(row.source_leaf_ids))
        require(len(key)==1 and key not in output_terms and row.output_consequent in VOCAB,
                "Invalid output-term rule mapping")
        output_terms[key]=row.output_consequent
    rules=[]
    for row in frozen_rules.itertuples(index=False):
        key=tuple(json.loads(row.source_leaf_ids))
        require(key in output_terms,"Unmatched source CART leaf")
        antecedent=json.loads(row.antecedent_or_groups_json)
        activation=infer_rule_strength(antecedent,values,partitions)
        readable_parts=[]
        for feature,terms in antecedent:
            readable_parts.append(NAME_BY_FEATURE[feature] + " is " + " OR ".join(terms))
        rules.append({
            "leaf": int(key[0]), "activation": activation,
            "consequent": output_terms[key],
            "antecedent": "IF " + " AND ".join(readable_parts) +
                " THEN difficulty term " + output_terms[key],
        })
    require(len(rules)==8 and len({r["leaf"] for r in rules})==8,
            "Missing or duplicated CART-derived Mamdani rule")
    pred,grid,agg=fuzzy_output(rules,centres)
    return rules,pred,grid,agg


def build_shap_comparison(
    development: pd.DataFrame, coefficients: dict[str, float],
    legacy: list[dict],
) -> dict:
    """Export only development observations, compared with each fixed stage reference.

    Excludes participant IDs and never uses the selected reserved-test cases as a
    reference. The SHAP coordinates are fixed linear-model attributions.
    """
    require(len(development) == 104 and
            development["participant_id"].nunique() == 26,
            "SHAP comparison requires 26 development participants and 104 trials")
    require(not development.duplicated(KEYS).any(),
            "Duplicate development trial in SHAP comparison")
    require(all(set(part["difficulty_level"].astype(int)) == {0, 2, 6, 10}
                and len(part) == 4
                for _, part in development.groupby("participant_id")),
            "Development participants must have four eligible stages")
    require(set(coefficients) == {"Intercept", *FEATURES},
            "SHAP comparison requires original frozen Auditory coefficients")
    raw = development.loc[:, FEATURES].to_numpy(dtype=float)
    require(np.isfinite(raw).all(), "Nonfinite development feature in SHAP comparison")
    fitted = coefficients["Intercept"] + sum(
        coefficients[key] * raw[:, j] for j, key in enumerate(FEATURES))
    require(np.isfinite(fitted).all(), "Nonfinite development model prediction")
    stage_codes = development["difficulty_level"].astype(int).tolist()
    output = {}
    for stage in ("0", "2", "6", "10"):
        mask = np.asarray([str(code) == stage for code in stage_codes])
        require(int(mask.sum()) == 26,
                "SHAP comparison requires 26 development trials at stage " + stage)
        means = raw[mask].mean(axis=0)
        baseline = float(fitted[mask].mean())
        reconstructed_baseline = coefficients["Intercept"] + sum(
            coefficients[key] * means[j] for j, key in enumerate(FEATURES))
        require(np.isclose(baseline, reconstructed_baseline,
                           atol=1e-10, rtol=0),
                "SHAP stage reference does not match frozen linear model")
        references = [
            numeric(r["Regression"]["ReferencePrediction"], "original stage reference")
            for r in legacy if r["DifficultyStage"] == STAGE_TEXT[stage]
        ]
        require(len(references) == 2 and
                all(np.isclose(baseline, value, atol=1e-7, rtol=0)
                    for value in references),
                "SHAP reference differs from the original MOD-16 waterfall")
        contributions = (raw - means[None, :]) * np.asarray(
            [coefficients[key] for key in FEATURES], dtype=float)[None, :]
        require(np.allclose(baseline + contributions.sum(axis=1), fitted,
                            atol=1e-8, rtol=0),
                "SHAP comparison failed frozen prediction reconstruction")
        output[stage] = {
            "referencePrediction": baseline,
            "meanFeatures": means.tolist(),
            "observations": [
                {"stage": str(code), "values": values.tolist(),
                 "contributions": phi.tolist()}
                for code, values, phi in zip(stage_codes, raw, contributions)
            ],
        }
    return {
        "schema": "auditory-shap-comparison-v1",
        "population": "development",
        "model": "auditory",
        "developmentParticipants": 26,
        "developmentObservations": 104,
        "features": [NAME_BY_FEATURE[key] for key in FEATURES],
        "byReferenceStage": output,
    }


def combined_payload(args: argparse.Namespace) -> dict:
    legacy=load_legacy(args.legacy_json)
    prior=assemble(args)  # Validates all existing development-only output sources.
    modelling=pd.read_csv(args.modeling_data)
    split=pd.read_csv(args.holdout_split)
    dev,test,dev_ids,test_ids=MOD14.prepare_split_data(
        modeling_data=modelling,holdout_split=split
    )
    require(set(IDS).issubset(set(test_ids)) and set(IDS).isdisjoint(dev_ids),
            "P06/P07 do not belong to the frozen reserved-test set")
    test_frame=MOD14.prepare_model_frame(test,model_id="auditory")
    test_frame=test_frame.loc[test_frame["participant_id"].isin(IDS)].copy()
    require(len(test_frame)==8 and test_frame["participant_id"].nunique()==2,
            "Expected exactly eight eligible test observations for P06/P07")
    test_frame=test_frame.sort_values(KEYS).reset_index(drop=True)
    regression=frozen_regression_coefficients(
        args.mod14_output_dir/"regression_coefficients.csv")
    partitions=load_saved_partitions(
        args.mamdani_fuzzy_dir/"tree_aligned_membership_partitions.csv")
    frozen_rules=pd.read_csv(args.mamdani_fuzzy_dir/"tree_aligned_fuzzy_rules.csv")
    frozen_mapping=pd.read_csv(args.mamdani_dir/"consequent_mapping.csv")
    output_info=json.loads(
        (args.mamdani_dir/"output_term_manifest.json").read_text(encoding="utf-8")
    )
    centres=np.asarray(
        output_info["output_vocabulary"]["output_5_terms"]["centres"],dtype=float
    )
    require(len(centres)==5 and np.all(np.diff(centres)>0),
            "Invalid frozen five-term Mamdani output universe")
    profiles=prior["profiles"]
    maxima=np.asarray([v["maximum"] for v in prior["features"]],dtype=float)
    scaled_centres=np.asarray([v["centre"] for v in profiles],dtype=float)/maxima
    dev_frame=MOD14.prepare_model_frame(dev,model_id="auditory")
    dev_frame=dev_frame.sort_values(KEYS).reset_index(drop=True)
    dev_scaled=dev_frame[list(FEATURES)].to_numpy(dtype=float)/maxima
    means=dev_scaled.mean(axis=0)
    _,_,vt=np.linalg.svd(dev_scaled-means,full_matrices=False)
    saved_points=prior["clusterMap"]["points"]
    require(np.allclose((dev_scaled-means)@vt[:2].T,
                        np.asarray([[p["x"],p["y"]] for p in saved_points]),
                        atol=1e-9,rtol=0),
            "Display PCA projection changed from development-only fit")
    shap_comparison = build_shap_comparison(dev_frame, regression, legacy)
    raw_original={(r["ParticipantId"],next(k for k,v in STAGE_TEXT.items()
                   if v==r["DifficultyStage"])):r for r in legacy}
    selected_points=[]
    cases=[]
    for _,record in test_frame.iterrows():
        who=str(record["participant_id"])
        stage=str(int(record["difficulty_level"]))
        key=(who,stage)
        require(key in raw_original,"No matching original A/B observation")
        original=raw_original[key]
        observed=original["Regression"]["Measurements"]
        obs_by_name={r["Name"]:numeric(r["Observed"], "measurement") for r in observed}
        measured=np.asarray([record[feature] for feature in FEATURES],dtype=float)
        for j,feature in enumerate(FEATURES):
            require(np.isclose(measured[j],obs_by_name[NAME_BY_FEATURE[feature]],
                               atol=1e-7,rtol=0),
                    "A/B feature differs from canonical P06/P07 Auditory test data")
        mlr=regression["Intercept"]+sum(regression[k]*measured[i]
                                      for i,k in enumerate(FEATURES))
        require(np.isclose(mlr,numeric(original["Regression"]["Prediction"], "original regression prediction"),
                           atol=1e-6,rtol=0),
                "A/B original prediction differs from frozen Auditory MLR")
        leaf=tree_leaf(prior["tree"],measured)
        mapped_values=dict(zip(FEATURES,measured.tolist()))
        mrules,mpred,grid,aggregated=infer_fuzzy_rules(
            partitions,frozen_rules,frozen_mapping,mapped_values,centres)
        scaled=measured/maxima
        u=membership_matrix(scaled[None,:],scaled_centres)[0]
        constants=np.asarray([p["constant"] for p in profiles],dtype=float)
        parts=u*constants
        sugeno=float(parts.sum())
        coord=(scaled-means)@vt[:2].T
        index=len(selected_points)
        selected_points.append({
            "x":float(coord[0]),"y":float(coord[1]),
            "memberships":u.tolist(),
        })
        cases.append({
            "example":who,"stage":stage,
            "originalPrediction":float(mlr),
            "measurements":measured.tolist(),
            "mapPointIndex":index,
            "mamdani":{
                "status":"OK" if mpred is not None else "Unavailable",
                "prediction":mpred,"outputTerms":5,
                "crispLeaf":int(leaf["leaf"]),
                "rules":sorted(mrules,key=lambda r:r["leaf"]),
                "outputCurve":{"x":grid.tolist(),"y":aggregated.tolist()}
                              if mpred is not None else None,
            },
            "sugeno":{
                "status":"OK","prediction":sugeno,
                "memberships":u.tolist(),"contributions":parts.tolist(),
            },
        })
    require(len(selected_points)==len(cases)==8,
            "Incorrect P06/P07 exploratory coverage")
    cd=copy.deepcopy(prior)
    cd["schema"]="auditory-cd-exploratory-v1"
    cd["population"]="reserved_test_exploratory"
    cd["testParticipantsDisplayed"]=2
    cd["testParticipantsUsedForFitting"]=0
    cd["cases"]=cases
    cd["clusterMap"]["selectedPoints"]=selected_points
    cd["limitations"]=[
        "P06 and P07 were already examined in the A/B interface; these are exploratory test-case demonstrations, not untouched independent validation.",
        "Frozen CART/Mamdani and FCM/Sugeno parameters were fitted only on 26 development participants.",
        "PCA overview uses the 104 development observations and projects test examples without refitting.",
        "SHAP explains the frozen regression; fuzzy approximations explain numerical difficulty only.",
        "No accuracy/test performance metrics, test-based model choice or adaptation policy are calculated.",
    ]
    return {
        "schema":"auditory-abcd-unified-exploratory-v1",
        "scientificStatus":"exploratory_previously_inspected_reserved_test_cases",
        "participants":["P06","P07"],
        "records":legacy,
        "cd":cd,
        "shapComparison":shap_comparison,
        "provenance":{
            "legacyViewerSha256":source_hash(args.legacy_json),
            "modelingDataSha256":source_hash(args.modeling_data),
            "holdoutSplitSha256":source_hash(args.holdout_split),
            "coefficientsSha256":source_hash(
                args.mod14_output_dir/"regression_coefficients.csv"),
            "trainingSources":prior["provenance"],
        },
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for item in ("legacy_json","modeling_data","holdout_split","mod14_output_dir",
                 "mamdani_tree_dir","mamdani_dir","mamdani_activation_dir",
                 "mamdani_fuzzy_dir","fcm_stage1_dir","sugeno_dir","output"):
        p.add_argument("--"+item.replace("_","-"),required=True,type=Path)
    args=p.parse_args()
    require(not args.output.exists(),"Refusing to overwrite existing local bundle")
    result=combined_payload(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,separators=(",",":"))+"\n",
                           encoding="utf-8")
    print("Unified A/B/C/D JSON created locally:",args.output)
    print("P06/P07: 8 matching cases, four methods, development-fitted fuzzy models.")
    print("Status: EXPLORATORY; previously inspected reserved-test participants.")
    print("Test metrics computed: NONE; models refitted: NONE.")


if __name__=="__main__":
    main()
