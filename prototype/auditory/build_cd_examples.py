#!/usr/bin/env python3
"""Build a LOCAL C/D viewer bundle from frozen Auditory development outputs.

No reserved-test case is read into the output. No model is trained/changed:
PCA is fitted solely for a labelled two-dimensional display of four-dimensional
development inputs. The saved FCM memberships and Mamdani rule strengths remain
the authoritative explanations. Never commit generated participant-level JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import MOD_14_direct_linear_regression_evaluation as MOD14

FEATURES = tuple(MOD14.AUDITORY_PREDICTORS)
KEYS = ["participant_id", "difficulty_level", "difficulty_stage"]
LEVELS = [(0, 0), (2, 1), (6, 2), (10, 3)]
VOCAB = ("Very low", "Low", "Medium", "High", "Very high")


class ExportError(ValueError):
    """Incomplete or altered development-source data."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ExportError(message)


def source(path: Path) -> pd.DataFrame:
    require(path.is_file(), "Missing source: " + str(path))
    table = pd.read_csv(path, dtype={"participant_id": str})
    require("participant_id" in table.columns, "Source lacks participant_id: " + str(path))
    return table


def align(table: pd.DataFrame, baseline: pd.DataFrame, label: str) -> pd.DataFrame:
    required = set(KEYS)
    require(required.issubset(table), label + ": missing participant/stage columns")
    require(len(table) == 104 and not table.duplicated(KEYS).any(),
            label + ": expected 104 unique development observations")
    left = table.copy()
    left["participant_id"] = left["participant_id"].astype(str)
    right = baseline.loc[:, KEYS].copy()
    result = right.merge(left, on=KEYS, how="left", validate="one_to_one", indicator=True)
    require(result["_merge"].eq("both").all(), label + ": changed participant/stage identities")
    return result.drop(columns="_merge")


def make_tree(numerical: pd.DataFrame) -> dict:
    require(len(numerical) == 8, "Expected eight CART numerical leaves")
    root = {}
    for _, record in numerical.iterrows():
        node = root
        conditions = json.loads(record["conditions_json"])
        for condition in conditions:
            require(condition["feature"] in FEATURES and
                    condition["operator"] in ("<=", ">"), "Invalid source-tree condition")
            feature = condition["feature"]
            threshold = float(condition["threshold"])
            if node:
                require(node.get("feature") == feature and
                        np.isclose(node.get("threshold"), threshold, atol=1e-12, rtol=0),
                        "Inconsistent CART path split")
            else:
                node.update({"feature": feature, "threshold": threshold,
                             "left": {}, "right": {}})
            node = node["left" if condition["operator"] == "<=" else "right"]
        require(not node, "Duplicate CART leaf")
        node.update({"leaf": int(record["leaf_id"]),
                     "value": float(record["tree_output"])})
    def check(node: dict) -> int:
        if "leaf" in node:
            return 1
        require(node.get("left") and node.get("right"), "Incomplete CART tree topology")
        return check(node["left"]) + check(node["right"])
    require(check(root) == 8, "Expected complete eight-leaf tree")
    return root


def tree_leaf(node: dict, values: np.ndarray) -> dict:
    while "leaf" not in node:
        idx = FEATURES.index(node["feature"])
        node = node["left" if values[idx] <= node["threshold"] else "right"]
    return node


def fuzzy_output(rules: list[dict], centres: np.ndarray) -> tuple[float | None, np.ndarray, np.ndarray]:
    grid = np.linspace(float(centres[0]), float(centres[-1]), 4001)
    terms = np.zeros((len(grid), len(centres)), dtype=float)
    for j in range(len(centres)):
        if j == 0:
            terms[:, j] = np.clip((centres[1] - grid) / (centres[1] - centres[0]), 0, 1)
        elif j == len(centres) - 1:
            terms[:, j] = np.clip((grid - centres[-2]) / (centres[-1] - centres[-2]), 0, 1)
        else:
            terms[:, j] = np.maximum(0, np.minimum(
                (grid - centres[j - 1]) / (centres[j] - centres[j - 1]),
                (centres[j + 1] - grid) / (centres[j + 1] - centres[j])
            ))
    combined = np.zeros(len(grid), dtype=float)
    for rule in rules:
        require(rule["consequent"] in VOCAB, "Unrecognised output term")
        combined = np.maximum(combined, np.minimum(
            float(rule["activation"]), terms[:, VOCAB.index(rule["consequent"])]
        ))
    widths = np.diff(grid)
    area = np.sum(widths * (combined[:-1] + combined[1:]) / 2)
    if area <= 1e-12:
        return None, grid, combined
    moment = np.sum(widths * (grid[:-1] * combined[:-1] +
                               grid[1:] * combined[1:]) / 2)
    return float(moment / area), grid, combined


def membership_matrix(x: np.ndarray, centres: np.ndarray) -> np.ndarray:
    squared = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    out = np.zeros_like(squared)
    for i, distances in enumerate(squared):
        zeros = distances == 0
        if zeros.any():
            out[i, zeros] = 1 / zeros.sum()
        else:
            inv = distances.min() / distances
            out[i] = inv / inv.sum()
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(args: argparse.Namespace) -> dict:
    split = pd.read_csv(args.holdout_split)
    modelling = pd.read_csv(args.modeling_data)
    development, _, dev_ids, test_ids = MOD14.prepare_split_data(
        modeling_data=modelling, holdout_split=split)
    frame = MOD14.prepare_model_frame(development, model_id="auditory")
    require(len(frame) == 104 and frame["participant_id"].nunique() == 26,
            "Canonical development Auditory frame must contain 26 participants x 4 stages")
    require(set(frame["participant_id"]) == set(dev_ids) and
            set(dev_ids).isdisjoint(test_ids), "Frozen development identities do not match")
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    require(all(set(zip(part["difficulty_level"], part["difficulty_stage"])) ==
                set(LEVELS) for _, part in frame.groupby("participant_id")),
            "Incomplete difficulty stages")

    tree_dir = args.mamdani_tree_dir
    mamdani_dir = args.mamdani_dir
    activation_dir = args.mamdani_activation_dir
    fcm_dir = args.fcm_stage1_dir
    sugeno_dir = args.sugeno_dir
    manifests = [
        (tree_dir / "manifest.json", "test_participants_used", "1: crisp tree extraction only"),
        (mamdani_dir / "output_term_manifest.json", "test_participants_used",
         "4A: output vocabulary sensitivity"),
        (activation_dir / "mamdani_manifest.json", "test_participants_used",
         "3: fixed Mamdani development baseline"),
        (fcm_dir / "cluster_manifest.json", "reserved_test_participants_used", None),
        (sugeno_dir / "zero_order_manifest.json", "reserved_test_participants_used",
         "2B: exploratory zero-order FCM-Sugeno"),
    ]
    provenance = {}
    for path, test_field, stage in manifests:
        require(path.is_file(), "Missing manifest: " + str(path))
        doc = json.loads(path.read_text(encoding="utf-8"))
        require(doc.get(test_field) == 0 and doc.get("development_participants") == 26,
                "Not a development-only source: " + path.name)
        if stage is not None:
            require(doc.get("stage") == stage, "Unexpected source stage: " + path.name)
        provenance[path.name + ":" + path.parent.name] = sha256(path)

    mamdani_meta = json.loads((mamdani_dir / "output_term_manifest.json").read_text())
    centres = np.asarray(mamdani_meta["output_vocabulary"]["output_5_terms"]["centres"], dtype=float)
    require(len(centres) == 5 and np.all(np.diff(centres) > 0), "Invalid five-term output centres")
    fcm_meta = json.loads((fcm_dir / "cluster_manifest.json").read_text())
    maxima = np.asarray([fcm_meta["training_feature_maxima"][k] for k in FEATURES], dtype=float)
    require(np.all(maxima > 0), "Invalid development scaling maxima")

    numeric_rules = pd.read_csv(tree_dir / "numerical_rules.csv")
    tree = make_tree(numeric_rules)
    mamdani = align(source(mamdani_dir / "output_term_predictions.csv"), frame, "Mamdani")
    sugeno = align(source(sugeno_dir / "development_predictions.csv"), frame, "Sugeno")
    require(np.allclose(mamdani["frozen_mlr_prediction"],
                        sugeno["frozen_mlr_prediction"], atol=1e-8, rtol=0),
            "Original Auditory regression teacher differs between models")
    mapping = pd.read_csv(mamdani_dir / "consequent_mapping.csv")
    five = mapping.loc[mapping["configuration"].eq("output_5_terms")]
    require(len(five) == 8 and five["source_leaf_ids"].nunique() == 8,
            "Incomplete five-term leaf-to-consequent map")
    consequent = {tuple(json.loads(row.source_leaf_ids)): row.output_consequent
                  for row in five.itertuples(index=False)}
    fuzzy_rules = pd.read_csv(args.mamdani_fuzzy_dir / "tree_aligned_fuzzy_rules.csv")
    antecedent = {
        tuple(json.loads(row.source_leaf_ids)): json.loads(row.antecedent_or_groups_json)
        for row in fuzzy_rules.itertuples(index=False)
    }
    activations = source(activation_dir / "rule_activations.csv")
    require(len(activations) == 104 * 8 and not activations[KEYS+["source_leaf_ids"]].duplicated().any(),
            "Expected eight unique rule activations for each of 104 observations")
    arules: dict[tuple[str, int, int], list[dict]] = {}
    for row in activations.itertuples(index=False):
        key = (str(row.participant_id), int(row.difficulty_level), int(row.difficulty_stage))
        leaves = tuple(json.loads(row.source_leaf_ids))
        require(len(leaves) == 1 and leaves in consequent and leaves in antecedent,
                "Unknown Mamdani source leaf")
        conditions = antecedent[leaves]
        names = {"mental_demand_score_0_to_10":"Mental demand",
                 "median_time_between_qualifying_grabs_seconds":"Time between grabs",
                 "median_time_to_target_seconds":"Time to target",
                 "median_head_turning_degrees":"Head turning"}
        clauses = []
        for feature, options in conditions:
            require(feature in names, "Unknown fuzzy antecedent feature")
            clauses.append(names[feature] + " is " + " OR ".join(options))
        text = "IF " + " AND ".join(clauses) + " THEN difficulty term " + consequent[leaves]
        arules.setdefault(key, []).append({
            "leaf": leaves[0], "activation": float(row.activation),
            "consequent": consequent[leaves], "antecedent": text
        })
    require(len(arules) == 104 and all(len(v) == 8 for v in arules.values()),
            "Incomplete activation-to-development joins")

    profile_table = pd.read_csv(sugeno_dir / "zero_order_rules.csv")
    require(profile_table["cluster"].tolist() == [1, 2, 3],
            "FCM cluster ordering changed")
    profiles = [{
        "cluster": int(row.cluster),
        "constant": float(row.constant_difficulty_stage),
        "centre": [float(getattr(row, f + "_centre_original_units")) for f in FEATURES],
    } for row in profile_table.itertuples(index=False)]
    original_centres = np.asarray([p["centre"] for p in profiles], dtype=float)
    scaled_centres = original_centres / maxima
    x = frame[list(FEATURES)].to_numpy(dtype=float)
    require(np.allclose(x.max(axis=0), maxima, rtol=0, atol=1e-8),
            "Saved FCM maxima differ from canonical development features")
    scaled = x / maxima
    saved_u = sugeno[[f"cluster_{i}_membership" for i in range(1, 4)]].to_numpy(dtype=float)
    require(np.allclose(membership_matrix(scaled, scaled_centres), saved_u,
                        atol=1e-7, rtol=0),
            "FCM memberships do not match saved four-dimensional centres")
    centered = scaled - scaled.mean(axis=0)
    _,singular,vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T
    pc_centres = (scaled_centres - scaled.mean(axis=0)) @ vt[:2].T
    explained = singular ** 2 / np.sum(singular ** 2)
    points = [{"x": float(a), "y": float(b), "memberships": list(map(float,u))}
              for (a,b),u in zip(projected,saved_u)]
    example_ids = sorted(frame["participant_id"].unique())[:2]
    require(len(example_ids) == 2, "Missing development examples")

    cases = []
    for person_index, participant in enumerate(example_ids,1):
        for level,stage in LEVELS:
            chosen = frame.loc[(frame["participant_id"] == participant) &
                               (frame["difficulty_level"] == level) &
                               (frame["difficulty_stage"] == stage)]
            require(len(chosen) == 1, "Missing example stage")
            index = int(chosen.index[0])
            mr, sr = mamdani.iloc[index], sugeno.iloc[index]
            values = x[index]
            leaf = tree_leaf(tree, values)
            require(np.isclose(float(mr["crisp_tree_prediction"]), leaf["value"],
                               atol=1e-8, rtol=0), "Saved crisp CART leaf does not reconstruct")
            key = (str(participant),int(level),int(stage))
            rules = sorted(arules[key], key=lambda r:r["leaf"])
            reconstructed, grid, output = fuzzy_output(rules, centres)
            status = str(mr["output_5_terms_status"])
            require(status in ("OK","Unavailable"), "Unknown Mamdani status")
            saved = (float(mr["output_5_terms_prediction"]) if status == "OK" else None)
            require((status == "Unavailable" and reconstructed is None) or
                    (status == "OK" and reconstructed is not None and
                     np.isclose(reconstructed,saved,atol=1e-7,rtol=0)),
                    "Five-term Mamdani centroid differs from saved result")
            memberships = [float(sr[f"cluster_{i}_membership"]) for i in (1,2,3)]
            contribution = [float(sr[f"rule_{i}_weighted_contribution"]) for i in (1,2,3)]
            predicted = float(sr["zero_order_sugeno_prediction"])
            require(np.isclose(sum(contribution), predicted, atol=1e-8, rtol=0),
                    "Sugeno weighted score failed reconstruction")
            cases.append({
                "example": "Development example " + str(person_index),
                "stage": str(level),
                "originalPrediction": float(mr["frozen_mlr_prediction"]),
                "measurements": [float(v) for v in values],
                "mapPointIndex": index,
                "mamdani": {
                    "status": status, "prediction": saved, "outputTerms": 5,
                    "crispLeaf": int(leaf["leaf"]), "rules": rules,
                    "outputCurve": {"x": grid.tolist(), "y": output.tolist()} if status == "OK" else None,
                },
                "sugeno": {
                    "status": "OK", "prediction": predicted, "memberships": memberships,
                    "contributions": contribution
                }
            })
    return {
        "schema": "auditory-cd-development-v1",
        "population": "development",
        "reservedTestParticipantsUsed": 0,
        "participants": 26, "observations": 104,
        "features": [{"key": feature, "maximum": float(maximum)}
                     for feature,maximum in zip(FEATURES,maxima)],
        "tree": tree, "profiles": profiles,
        "clusterMap": {
            "type": "pca-development",
            "explainedVariance": [float(v) for v in explained[:2]],
            "points": points,
            "centres": [{"x": float(a), "y": float(b)} for a,b in pc_centres]
        },
        "cases": cases,
        "provenance": provenance,
        "limitations": [
            "Tree path is the crisp CART route; other Mamdani rules may activate.",
            "PCA coordinates are a display-only projection and do not determine memberships.",
            "The numerical fuzzy outputs imitate the original development-fitted Auditory MLR.",
            "No adaptation recommendation or clinical calibration is implied.",
            "The existing A/B demonstration uses reserved test cases and must not be treated as equivalent development-case comparisons."
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("modeling_data", "holdout_split", "mamdani_tree_dir",
                 "mamdani_dir", "mamdani_activation_dir", "mamdani_fuzzy_dir",
                 "fcm_stage1_dir", "sugeno_dir", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite existing local JSON: " + str(args.output))
    payload = assemble(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,separators=(",",":"))+"\n",
                           encoding="utf-8")
    print("Created LOCAL development bundle:",args.output)
    print("Development participants:",payload["participants"],
          "| displayed examples:",len(payload["cases"]),
          "| reserved-test observations used:",payload["reservedTestParticipantsUsed"])


if __name__ == "__main__":
    main()
