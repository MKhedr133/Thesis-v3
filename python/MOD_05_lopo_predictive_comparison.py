#!/usr/bin/env python3
"""MOD-05 — Leave-one-participant-out predictive comparison.

Every candidate fixed-effect model is evaluated under BOTH retained random-
effects structures:

    RI     : random intercept only
    RI_RS  : random intercept + random difficulty slope

TMT-B is standardized from unique training participants inside each fold. Held-
out prediction is population/fixed-effect only; no held-out participant random
effect is estimated from test outcomes. The RI versus RI+RS structure is not selected here.

Runtime strategy
----------------
The expensive unit is one condition × target × random-structure family. Each
family contains 32 held-out folds and all candidate fixed-effect models. Families
can run in parallel across processes. A JSON checkpoint is updated after every
held-out participant so an interrupted run can safely resume without refitting
completed folds when the data/configuration/code signature still matches.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Mapping

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_RANDOM_STRUCTURES = ("RI", "RI_RS")
CHECKPOINT_SCHEMA_VERSION = 2
DEFAULT_AUTO_MAX_JOBS = 4


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


MOD02 = _load("mod02_for_lopo", "MOD_02_random_effects_structure.py")
MOD04 = _load("mod04_for_lopo", "MOD_04_age_tmt_associations.py")


@dataclass(frozen=True)
class LopoResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    predictions: pd.DataFrame
    participant_errors: pd.DataFrame
    model_summary: pd.DataFrame
    checks: pd.DataFrame


@dataclass(frozen=True)
class FamilyTask:
    condition: str
    target: str
    random_structure: str
    difficulty_column: str
    frame: pd.DataFrame
    registry_items: tuple[tuple[str, str], ...]
    signature: str
    checkpoint_path: str | None
    resume: bool


@dataclass(frozen=True)
class FamilyResult:
    condition: str
    target: str
    random_structure: str
    prediction_rows: tuple[dict[str, object], ...]
    participant_rows: tuple[dict[str, object], ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    family_failures: int
    resumed_from_checkpoint: bool


def _registry(
    registry: Mapping[str, str] | None,
    use_proposed_registry: bool,
) -> dict[str, str]:
    if registry is not None:
        return dict(registry)
    if use_proposed_registry:
        return dict(MOD04.PROPOSED_MODEL_RHS)
    raise ValueError("No approved candidate-model registry supplied")


def _difficulty_columns() -> dict[str, str]:
    return {
        target: spec.difficulty_column
        for target, spec in MOD02.build_internal_registry().items()
    }


def _predict_fixed_effects(result, test: pd.DataFrame) -> np.ndarray:
    """Formula MixedLM ``predict`` uses fixed effects only for new rows."""
    return np.asarray(result.predict(test), dtype=float)


def resolve_job_count(requested: int, *, cpu_count: int | None = None) -> int:
    """Resolve ``0`` to a conservative automatic process count.

    MixedLM optimization is CPU-heavy and numerical libraries may also use
    threads internally, so automatic mode is deliberately capped at four worker
    processes. An explicit positive ``--jobs`` value is respected.
    """
    if requested < 0:
        raise ValueError("jobs must be >= 0; use 0 for automatic selection")
    if requested > 0:
        return requested
    available = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    return max(1, min(DEFAULT_AUTO_MAX_JOBS, available))


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "item"


def family_checkpoint_path(
    root: Path,
    condition: str,
    target: str,
    random_structure: str,
) -> Path:
    """Return the deterministic checkpoint filename for one model family."""
    name = "__".join(
        [_safe_slug(condition), _safe_slug(target), _safe_slug(random_structure)]
    )
    return Path(root) / f"{name}.json"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_family_checkpoint(path: Path, payload: dict[str, object]) -> None:
    """Atomically save a family checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=True, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temp, path)


def load_family_checkpoint(
    path: Path,
    *,
    expected_signature: str,
) -> dict[str, object] | None:
    """Load a checkpoint only when its schema/signature match this run."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") not in (None, CHECKPOINT_SCHEMA_VERSION):
        return None
    if payload.get("signature") != expected_signature:
        return None
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pipeline_code_signature() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(MOD02.__file__), Path(MOD04.__file__)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _family_signature(
    frame: pd.DataFrame,
    *,
    condition: str,
    target: str,
    random_structure: str,
    difficulty_column: str,
    registry: Mapping[str, str],
    code_signature: str,
) -> str:
    """Create a deterministic signature for checkpoint validation."""
    digest = hashlib.sha256()
    config = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "condition": condition,
        "target": target,
        "random_structure": random_structure,
        "difficulty_column": difficulty_column,
        "registry": list(registry.items()),
        "code_signature": code_signature,
    }
    digest.update(json.dumps(config, sort_keys=True).encode("utf-8"))

    columns = list(
        dict.fromkeys(
            [
                "participant_id",
                "participant_group",
                "tmt_b_seconds",
                "difficulty_level",
                difficulty_column,
                target,
            ]
        )
    )
    stable = frame[columns].copy()
    sort_columns = [c for c in ("participant_id", "difficulty_level") if c in stable]
    if sort_columns:
        stable = stable.sort_values(sort_columns, kind="mergesort")
    digest.update(stable.to_csv(index=False, float_format="%.17g").encode("utf-8"))
    return digest.hexdigest()


def _checkpoint_payload(
    task: FamilyTask,
    *,
    completed_participants: list[str],
    prediction_rows: list[dict[str, object]],
    participant_rows: list[dict[str, object]],
    errors: list[str],
    warnings_out: list[str],
    family_failures: int,
    complete: bool,
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "signature": task.signature,
        "condition_name": task.condition,
        "target_name": task.target,
        "random_structure": task.random_structure,
        "completed_participants": completed_participants,
        "prediction_rows": prediction_rows,
        "participant_rows": participant_rows,
        "errors": errors,
        "warnings": warnings_out,
        "family_failures": family_failures,
        "complete": complete,
    }


def _result_from_payload(task: FamilyTask, payload: Mapping[str, object]) -> FamilyResult:
    return FamilyResult(
        condition=task.condition,
        target=task.target,
        random_structure=task.random_structure,
        prediction_rows=tuple(payload.get("prediction_rows", [])),
        participant_rows=tuple(payload.get("participant_rows", [])),
        errors=tuple(str(x) for x in payload.get("errors", [])),
        warnings=tuple(str(x) for x in payload.get("warnings", [])),
        family_failures=int(payload.get("family_failures", 0)),
        resumed_from_checkpoint=True,
    )


def _run_family_task(task: FamilyTask) -> FamilyResult:
    """Fit all LOPO folds/models for one condition × target × structure family."""
    checkpoint_path = Path(task.checkpoint_path) if task.checkpoint_path else None
    checkpoint = None
    if task.resume and checkpoint_path is not None:
        checkpoint = load_family_checkpoint(
            checkpoint_path,
            expected_signature=task.signature,
        )
        if checkpoint is not None and bool(checkpoint.get("complete", False)):
            return _result_from_payload(task, checkpoint)

    if checkpoint is not None:
        completed_participants = [str(x) for x in checkpoint.get("completed_participants", [])]
        prediction_rows = list(checkpoint.get("prediction_rows", []))
        participant_rows = list(checkpoint.get("participant_rows", []))
        errors = [str(x) for x in checkpoint.get("errors", [])]
        warnings_out = [str(x) for x in checkpoint.get("warnings", [])]
        family_failures = int(checkpoint.get("family_failures", 0))
        resumed = bool(completed_participants)
    else:
        completed_participants = []
        prediction_rows = []
        participant_rows = []
        errors = []
        warnings_out = []
        family_failures = 0
        resumed = False

    completed_set = set(completed_participants)
    registry = dict(task.registry_items)
    frame = task.frame
    participants = tuple(sorted(frame["participant_id"].astype(str).unique()))

    for heldout in participants:
        if heldout in completed_set:
            continue

        test_mask = frame["participant_id"].astype(str).eq(heldout)
        train = frame.loc[~test_mask].copy()
        test = frame.loc[test_mask].copy()

        train_participant_tmt = train[
            ["participant_id", "tmt_b_seconds"]
        ].drop_duplicates("participant_id")
        tmt_values = pd.to_numeric(
            train_participant_tmt["tmt_b_seconds"], errors="coerce"
        )
        tmt_mean = float(tmt_values.mean())
        tmt_sd = float(tmt_values.std(ddof=1))

        if not np.isfinite(tmt_sd) or tmt_sd <= 0:
            errors.append(
                f"{task.condition}/{task.target}/{task.random_structure}/{heldout}: "
                "invalid training TMT SD"
            )
            family_failures += len(registry)
        else:
            train["tmt_z"] = (
                pd.to_numeric(train["tmt_b_seconds"], errors="coerce") - tmt_mean
            ) / tmt_sd
            test["tmt_z"] = (
                pd.to_numeric(test["tmt_b_seconds"], errors="coerce") - tmt_mean
            ) / tmt_sd

            for model_id, rhs_template in registry.items():
                formula = f"{task.target} ~ {rhs_template.format(D=task.difficulty_column)}"
                try:
                    result, optimizer, fit_warnings, fit_errors = (
                        MOD02.fit_mixedlm_with_fallback(
                            formula,
                            train,
                            task.random_structure,
                            task.difficulty_column,
                        )
                    )
                    warnings_out.extend(
                        f"{task.condition}/{task.target}/{task.random_structure}/"
                        f"{heldout}/{model_id}: {warning}"
                        for warning in fit_warnings
                    )
                    if result is None or not bool(getattr(result, "converged", False)):
                        raise RuntimeError(
                            "training fit failed/non-converged: " + " | ".join(fit_errors)
                        )

                    predictions = _predict_fixed_effects(result, test)
                    observed = pd.to_numeric(
                        test[task.target], errors="coerce"
                    ).to_numpy(float)
                    if len(predictions) != len(test):
                        raise RuntimeError("Prediction length mismatch")
                    abs_error = np.abs(observed - predictions)

                    for local_idx, (_, row) in enumerate(test.iterrows()):
                        prediction_rows.append({
                            "heldout_participant": heldout,
                            "condition_name": task.condition,
                            "target_name": task.target,
                            "model_id": model_id,
                            "random_structure": task.random_structure,
                            "difficulty_level": row.get("difficulty_level", np.nan),
                            "observed": observed[local_idx],
                            "predicted": predictions[local_idx],
                            "residual": observed[local_idx] - predictions[local_idx],
                            "absolute_error": abs_error[local_idx],
                            "training_participant_count": int(
                                train["participant_id"].nunique()
                            ),
                            "training_observation_count": int(len(train)),
                            "training_tmt_mean": tmt_mean,
                            "training_tmt_sd": tmt_sd,
                            "optimizer": optimizer,
                            "prediction_scope": "fixed_effect_population_only",
                        })

                    participant_rows.append({
                        "participant_id": heldout,
                        "condition_name": task.condition,
                        "target_name": task.target,
                        "model_id": model_id,
                        "random_structure": task.random_structure,
                        "valid_prediction_count": int(len(test)),
                        "participant_mae": float(np.mean(abs_error)),
                        "participant_rmse": float(
                            np.sqrt(np.mean((observed - predictions) ** 2))
                        ),
                        "fold_status": "pass",
                    })
                except Exception as exc:
                    family_failures += 1
                    errors.append(
                        f"{task.condition}/{task.target}/{task.random_structure}/"
                        f"{heldout}/{model_id}: {type(exc).__name__}: {exc}"
                    )
                    participant_rows.append({
                        "participant_id": heldout,
                        "condition_name": task.condition,
                        "target_name": task.target,
                        "model_id": model_id,
                        "random_structure": task.random_structure,
                        "valid_prediction_count": 0,
                        "participant_mae": np.nan,
                        "participant_rmse": np.nan,
                        "fold_status": "fail",
                    })

        completed_participants.append(heldout)
        completed_set.add(heldout)
        if checkpoint_path is not None:
            save_family_checkpoint(
                checkpoint_path,
                _checkpoint_payload(
                    task,
                    completed_participants=completed_participants,
                    prediction_rows=prediction_rows,
                    participant_rows=participant_rows,
                    errors=errors,
                    warnings_out=warnings_out,
                    family_failures=family_failures,
                    complete=False,
                ),
            )

    if checkpoint_path is not None:
        save_family_checkpoint(
            checkpoint_path,
            _checkpoint_payload(
                task,
                completed_participants=completed_participants,
                prediction_rows=prediction_rows,
                participant_rows=participant_rows,
                errors=errors,
                warnings_out=warnings_out,
                family_failures=family_failures,
                complete=True,
            ),
        )

    return FamilyResult(
        condition=task.condition,
        target=task.target,
        random_structure=task.random_structure,
        prediction_rows=tuple(prediction_rows),
        participant_rows=tuple(participant_rows),
        errors=tuple(errors),
        warnings=tuple(warnings_out),
        family_failures=family_failures,
        resumed_from_checkpoint=resumed,
    )


def _configure_worker_thread_limits() -> None:
    """Limit nested BLAS/OpenMP threading when using multiple worker processes."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def _print_progress(
    completed: int,
    total: int,
    *,
    started_at: float,
    resumed_families: int,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started_at)
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - completed)
    eta = remaining / rate if rate > 0 else float("nan")
    eta_text = f"{eta / 60:.1f} min" if np.isfinite(eta) else "estimating"
    print(
        f"[MOD-05] families {completed}/{total} | elapsed {elapsed / 60:.1f} min "
        f"| ETA {eta_text} | checkpoint-resumed {resumed_families}",
        flush=True,
    )


def run_lopo_prediction(
    modeling_data: pd.DataFrame,
    *,
    model_rhs_registry: Mapping[str, str] | None = None,
    use_proposed_registry: bool = False,
    targets: tuple[str, ...] = MOD02.GAUSSIAN_TARGETS,
    blocked_targets: tuple[str, ...] = MOD02.PENDING_COUNT_TARGETS,
    conditions: tuple[str, ...] = MOD02.CONDITIONS,
    random_structures: tuple[str, ...] = DEFAULT_RANDOM_STRUCTURES,
    difficulty_columns: Mapping[str, str] | None = None,
    jobs: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    progress: bool = False,
    progress_every: int = 1,
) -> LopoResult:
    """Run whole-participant LOPO independently for RI and RI+RS.

    ``jobs`` controls parallel condition × target × random-structure families.
    Set ``jobs=1`` for deterministic serial execution or ``jobs=0`` for a
    conservative automatic process count. Checkpoints are optional at API level;
    the CLI enables them by default under the output directory.
    """
    invalid = [s for s in random_structures if s not in {"RI", "RI_RS"}]
    if invalid:
        raise ValueError(f"Unsupported random structure(s): {invalid}")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")

    resolved_jobs = resolve_job_count(jobs)
    registry = _registry(model_rhs_registry, use_proposed_registry)
    dcols = dict(difficulty_columns or _difficulty_columns())
    data = MOD02.prepare_internal_modeling_data(modeling_data)
    code_signature = _pipeline_code_signature()

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[FamilyTask] = []
    check_rows: list[dict[str, object]] = []
    setup_errors: list[str] = []

    for condition in conditions:
        for target in targets:
            if target in blocked_targets:
                check_rows.append({
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": "",
                    "status": "blocked",
                    "message": "Response model family unresolved.",
                })
                continue

            dcol = dcols.get(target, "")
            if not dcol:
                setup_errors.append(f"{condition}/{target}: missing difficulty column")
                continue

            frame = data.loc[data["condition_name"].eq(condition)].copy()
            if target == "performance_change_from_d0_percentage_points":
                if dcol == "difficulty_stage":
                    setup_errors.append(
                        f"{condition}/{target}: performance cannot silently use difficulty_stage"
                    )
                    continue
                frame = frame.loc[~frame["difficulty_level"].eq(0)].copy()

            required = [
                "participant_id",
                "participant_group",
                "tmt_b_seconds",
                target,
                dcol,
            ]
            missing = [c for c in required if c not in frame.columns]
            if missing:
                setup_errors.append(f"{condition}/{target}: missing columns {missing}")
                continue
            frame = frame.loc[frame[required].notna().all(axis=1)].copy()
            participants = tuple(sorted(frame["participant_id"].astype(str).unique()))
            if len(participants) < 3:
                setup_errors.append(f"{condition}/{target}: too few participants for LOPO")
                continue

            for structure in random_structures:
                signature = _family_signature(
                    frame,
                    condition=condition,
                    target=target,
                    random_structure=structure,
                    difficulty_column=dcol,
                    registry=registry,
                    code_signature=code_signature,
                )
                checkpoint_path = (
                    family_checkpoint_path(
                        checkpoint_dir,
                        condition,
                        target,
                        structure,
                    )
                    if checkpoint_dir is not None
                    else None
                )
                tasks.append(
                    FamilyTask(
                        condition=condition,
                        target=target,
                        random_structure=structure,
                        difficulty_column=dcol,
                        frame=frame,
                        registry_items=tuple(registry.items()),
                        signature=signature,
                        checkpoint_path=(
                            str(checkpoint_path) if checkpoint_path is not None else None
                        ),
                        resume=resume,
                    )
                )

    started_at = time.perf_counter()
    family_results: list[FamilyResult] = []
    completed_count = 0
    resumed_count = 0

    if resolved_jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            result = _run_family_task(task)
            family_results.append(result)
            completed_count += 1
            resumed_count += int(result.resumed_from_checkpoint)
            if progress and (
                completed_count % progress_every == 0 or completed_count == len(tasks)
            ):
                _print_progress(
                    completed_count,
                    len(tasks),
                    started_at=started_at,
                    resumed_families=resumed_count,
                )
    else:
        _configure_worker_thread_limits()
        with ProcessPoolExecutor(max_workers=resolved_jobs) as executor:
            future_to_task = {
                executor.submit(_run_family_task, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = FamilyResult(
                        condition=task.condition,
                        target=task.target,
                        random_structure=task.random_structure,
                        prediction_rows=(),
                        participant_rows=(),
                        errors=(
                            f"{task.condition}/{task.target}/{task.random_structure}: "
                            f"worker failure: {type(exc).__name__}: {exc}",
                        ),
                        warnings=(),
                        family_failures=len(registry)
                        * int(task.frame["participant_id"].nunique()),
                        resumed_from_checkpoint=False,
                    )
                family_results.append(result)
                completed_count += 1
                resumed_count += int(result.resumed_from_checkpoint)
                if progress and (
                    completed_count % progress_every == 0
                    or completed_count == len(tasks)
                ):
                    _print_progress(
                        completed_count,
                        len(tasks),
                        started_at=started_at,
                        resumed_families=resumed_count,
                    )

    condition_order = {value: idx for idx, value in enumerate(conditions)}
    target_order = {value: idx for idx, value in enumerate(targets)}
    structure_order = {value: idx for idx, value in enumerate(random_structures)}
    family_results.sort(
        key=lambda result: (
            condition_order.get(result.condition, 10**6),
            target_order.get(result.target, 10**6),
            structure_order.get(result.random_structure, 10**6),
        )
    )

    prediction_rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []
    errors = list(setup_errors)
    warnings_out: list[str] = []

    for family in family_results:
        prediction_rows.extend(family.prediction_rows)
        participant_rows.extend(family.participant_rows)
        errors.extend(family.errors)
        warnings_out.extend(family.warnings)
        check_rows.append({
            "condition_name": family.condition,
            "target_name": family.target,
            "random_structure": family.random_structure,
            "status": "pass" if family.family_failures == 0 else "fail",
            "message": (
                f"LOPO completed with {family.family_failures} failed "
                "model-fold fit(s)."
            ),
            "resumed_from_checkpoint": family.resumed_from_checkpoint,
        })

    participant_errors = pd.DataFrame(participant_rows)
    summary_rows: list[dict[str, object]] = []
    if len(participant_errors):
        passed = participant_errors.loc[
            participant_errors["fold_status"].eq("pass")
        ].copy()
        group_keys = ["condition_name", "target_name", "random_structure"]
        for (condition, target, structure), family in passed.groupby(
            group_keys, sort=False
        ):
            model_mae = family.groupby("model_id")["participant_mae"].mean()
            m0 = model_mae.get("M0", np.nan)
            for model_id, mae in model_mae.items():
                model_rows = family.loc[family["model_id"].eq(model_id)]
                summary_rows.append({
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": structure,
                    "model_id": model_id,
                    "participant_count": int(
                        model_rows["participant_id"].nunique()
                    ),
                    "participant_balanced_mae": float(mae),
                    "participant_balanced_rmse": float(
                        model_rows["participant_rmse"].mean()
                    ),
                    "delta_mae_vs_m0": (
                        float(m0 - mae) if np.isfinite(m0) else np.nan
                    ),
                    "delta_mae_convention": (
                        "MAE_M0_minus_MAE_model; positive means candidate predicts better"
                    ),
                })

    return LopoResult(
        not errors,
        tuple(errors),
        tuple(warnings_out),
        pd.DataFrame(prediction_rows),
        participant_errors,
        pd.DataFrame(summary_rows),
        pd.DataFrame(check_rows),
    )


def write_lopo_outputs(result: LopoResult, output_dir: Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": out / "lopo_predictions.csv",
        "participant_errors": out / "lopo_participant_errors.csv",
        "summary": out / "lopo_model_summary.csv",
        "checks": out / "lopo_checks.csv",
    }
    result.predictions.to_csv(
        paths["predictions"], index=False, encoding="utf-8", na_rep=""
    )
    result.participant_errors.to_csv(
        paths["participant_errors"], index=False, encoding="utf-8", na_rep=""
    )
    result.model_summary.to_csv(
        paths["summary"], index=False, encoding="utf-8", na_rep=""
    )
    result.checks.to_csv(
        paths["checks"], index=False, encoding="utf-8", na_rep=""
    )
    return paths


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--model-registry-json", type=Path)
    parser.add_argument("--use-proposed-registry", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Parallel worker processes. 0=automatic (capped at 4); "
            "1=serial; positive values are used exactly."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Checkpoint directory; defaults to OUTPUT_DIR/_lopo_checkpoints.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable fold checkpoints/resume entirely.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching existing checkpoints and refit all folds.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress after this many completed model families.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress runtime progress/ETA messages.",
    )
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    jobs = resolve_job_count(args.jobs)
    checkpoint_dir = None
    if not args.no_checkpoint:
        checkpoint_dir = (
            args.checkpoint_dir
            if args.checkpoint_dir is not None
            else args.output_dir / "_lopo_checkpoints"
        )

    print(
        f"MOD-05 starting: jobs={jobs}; checkpointing="
        f"{'off' if checkpoint_dir is None else checkpoint_dir}; "
        f"resume={not args.no_resume}; random_structures=RI,RI_RS",
        flush=True,
    )

    result = run_lopo_prediction(
        pd.read_csv(args.modeling_data),
        model_rhs_registry=(
            _load_json(args.model_registry_json)
            if args.model_registry_json
            else None
        ),
        use_proposed_registry=args.use_proposed_registry,
        jobs=jobs,
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        progress=not args.no_progress,
        progress_every=args.progress_every,
    )
    write_lopo_outputs(result, args.output_dir)
    print(
        f"lopo_valid={result.valid}; errors={len(result.errors)}; "
        f"warnings={len(result.warnings)}; random_structures=RI,RI_RS",
        flush=True,
    )
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
