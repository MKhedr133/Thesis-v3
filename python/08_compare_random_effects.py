"""Compare MOD-02B RI and MOD-02C RI+RS fit evidence in memory."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd


def _load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load module {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


_REGISTRY_MODULE = _load_module(
    Path(__file__).with_name("05_compare_random_effects.py"),
    "mod02a_registry_for_mod02d",
)
_RI_MODULE = _load_module(
    Path(__file__).with_name("06_fit_random_intercept_models.py"),
    "mod02b_ri_for_mod02d",
)
_RS_MODULE = _load_module(
    Path(__file__).with_name("07_fit_random_intercept_plus_slope_models.py"),
    "mod02c_rs_for_mod02d",
)


RandomEffectsComparisonRegistry = _REGISTRY_MODULE.RandomEffectsComparisonRegistry
RandomInterceptFitResult = _RI_MODULE.RandomInterceptFitResult
RandomEffectsFitResult = _RS_MODULE.RandomEffectsFitResult
COMPARISON_EXTRA_FIELDS = _RI_MODULE.COMPARISON_EXTRA_FIELDS
LIKELIHOOD_RATIO_ROLE = _REGISTRY_MODULE.LIKELIHOOD_RATIO_ROLE
SELECTION_STATUS = _REGISTRY_MODULE.SELECTION_STATUS


@dataclass(frozen=True)
class RandomEffectsPairAudit:
    """Pair-level structural and observation-set audit."""

    condition_name: str
    target_name: str
    ri_record_count: int
    ri_rs_record_count: int
    ri_included_row_signature: str | None
    ri_rs_included_row_signature: str | None
    ri_observation_count: int | None
    ri_rs_observation_count: int | None
    ri_participant_count: int | None
    ri_rs_participant_count: int | None
    same_observations: bool
    status: str
    likelihood_ratio_statistic: float | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RandomEffectsComparisonResult:
    """Immutable comparison result; contained pandas frames remain mutable."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    pair_audits: tuple[RandomEffectsPairAudit, ...]
    comparison_table: pd.DataFrame


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _finite_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _record_key(record: Any) -> tuple[str, str]:
    return (record.condition_name, record.target_name)


def _records_by_key(records: tuple[Any, ...]) -> dict[tuple[str, str], list[Any]]:
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for record in records:
        grouped[_record_key(record)].append(record)
    return dict(grouped)


def _record_row(record: Any, registry: RandomEffectsComparisonRegistry) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in registry.comparison_fields:
        value = getattr(record, field)
        if field == "warnings":
            value = "; ".join(record.warnings)
        row[field] = value
    for field in COMPARISON_EXTRA_FIELDS:
        row[field] = getattr(record, field)
    return row


def _metadata_mismatches(ri_record: Any, rs_record: Any) -> list[str]:
    mismatches: list[str] = []
    if ri_record.model_id != "RI":
        mismatches.append(f"RI record has model_id={ri_record.model_id!r}")
    if rs_record.model_id != "RI_RS":
        mismatches.append(f"RI+RS record has model_id={rs_record.model_id!r}")
    if ri_record.random_structure != "RI":
        mismatches.append(
            f"RI record has random_structure={ri_record.random_structure!r}"
        )
    if rs_record.random_structure != "RI_RS":
        mismatches.append(
            f"RI+RS record has random_structure={rs_record.random_structure!r}"
        )

    matching_fields = (
        "condition_name",
        "target_name",
        "target_role",
        "response_coding",
        "fixed_effects_formula",
        "executable_formula",
        "fixed_effects_status",
        "difficulty_term",
        "difficulty_source_column",
        "difficulty_coding_status",
        "difficulty_mapping",
        "estimation_method",
        "reml",
        "participant_count",
        "observation_count",
        "excluded_observation_count",
        "excluded_participant_count",
        "included_row_signature",
    )
    for field in matching_fields:
        ri_value = getattr(ri_record, field)
        rs_value = getattr(rs_record, field)
        if ri_value != rs_value:
            mismatches.append(
                f"{field} differs between RI and RI_RS: "
                f"{ri_value!r} != {rs_value!r}"
            )
    return mismatches


def _convergence_errors(record: Any, label: str) -> list[str]:
    errors: list[str] = []
    if record.fit_status != "fitted":
        errors.append(
            f"{label} fit_status must be 'fitted', got {record.fit_status!r}"
        )
    if record.convergence_status is not True:
        errors.append(
            f"{label} convergence_status must be True, "
            f"got {record.convergence_status!r}"
        )
    return errors


def _metric_deltas(
    ri_record: Any,
    rs_record: Any,
    field: str,
    label: str,
    condition_name: str,
    target_name: str,
) -> tuple[float | None, float | None, str | None]:
    ri_value = _finite_float(getattr(ri_record, field))
    rs_value = _finite_float(getattr(rs_record, field))
    if ri_value is None or rs_value is None:
        return (
            None,
            None,
            f"{condition_name}/{target_name}: {label} unavailable for one or both "
            f"fits; derived {label.lower()} deltas remain missing.",
        )
    minimum = min(ri_value, rs_value)
    return ri_value - minimum, rs_value - minimum, None


def _likelihood_ratio(
    ri_record: Any,
    rs_record: Any,
    condition_name: str,
    target_name: str,
) -> tuple[float | None, str | None]:
    ri_value = _finite_float(ri_record.log_likelihood)
    rs_value = _finite_float(rs_record.log_likelihood)
    if ri_value is None or rs_value is None:
        return (
            None,
            f"{condition_name}/{target_name}: log likelihood unavailable for one "
            "or both fits; likelihood-ratio statistic remains missing.",
        )
    statistic = 2.0 * (rs_value - ri_value)
    warning = None
    if statistic < 0:
        warning = (
            f"{condition_name}/{target_name}: RI+RS log likelihood is below RI; "
            "the signed supporting likelihood-ratio statistic is preserved."
        )
    return statistic, warning


def _pair_audit(
    condition_name: str,
    target_name: str,
    ri_records: list[Any],
    rs_records: list[Any],
) -> tuple[RandomEffectsPairAudit, Any | None, Any | None, tuple[str, ...], tuple[str, ...]]:
    pair_errors: list[str] = []
    pair_warnings: list[str] = []
    ri_record = ri_records[0] if ri_records else None
    rs_record = rs_records[0] if rs_records else None

    if len(ri_records) != 1:
        pair_errors.append(
            f"{condition_name}/{target_name}: expected exactly one RI record, "
            f"found {len(ri_records)}"
        )
    if len(rs_records) != 1:
        pair_errors.append(
            f"{condition_name}/{target_name}: expected exactly one RI_RS record, "
            f"found {len(rs_records)}"
        )

    if ri_record is not None:
        pair_errors.extend(_convergence_errors(ri_record, "RI"))
    if rs_record is not None:
        pair_errors.extend(_convergence_errors(rs_record, "RI_RS"))

    if ri_record is not None and rs_record is not None:
        pair_errors.extend(_metadata_mismatches(ri_record, rs_record))

    ri_signature = getattr(ri_record, "included_row_signature", None)
    rs_signature = getattr(rs_record, "included_row_signature", None)
    ri_observations = getattr(ri_record, "observation_count", None)
    rs_observations = getattr(rs_record, "observation_count", None)
    ri_participants = getattr(ri_record, "participant_count", None)
    rs_participants = getattr(rs_record, "participant_count", None)
    same_observations = (
        ri_record is not None
        and rs_record is not None
        and ri_signature == rs_signature
        and ri_observations == rs_observations
    )
    if ri_record is not None and rs_record is not None and not same_observations:
        pair_errors.append(
            f"{condition_name}/{target_name}: RI and RI_RS do not use identical "
            "included observations"
        )

    statistic = None
    if not pair_errors and ri_record is not None and rs_record is not None:
        statistic, warning = _likelihood_ratio(
            ri_record,
            rs_record,
            condition_name,
            target_name,
        )
        if warning:
            pair_warnings.append(warning)

        for field, label in (("aic", "AIC"), ("bic", "BIC")):
            _, _, warning = _metric_deltas(
                ri_record,
                rs_record,
                field,
                label,
                condition_name,
                target_name,
            )
            if warning:
                pair_warnings.append(warning)

    audit = RandomEffectsPairAudit(
        condition_name=condition_name,
        target_name=target_name,
        ri_record_count=len(ri_records),
        ri_rs_record_count=len(rs_records),
        ri_included_row_signature=ri_signature,
        ri_rs_included_row_signature=rs_signature,
        ri_observation_count=ri_observations,
        ri_rs_observation_count=rs_observations,
        ri_participant_count=ri_participants,
        ri_rs_participant_count=rs_participants,
        same_observations=same_observations,
        status="comparable" if not pair_errors else "failed",
        likelihood_ratio_statistic=statistic,
        errors=_unique(pair_errors),
        warnings=_unique(pair_warnings),
    )
    return audit, ri_record, rs_record, audit.errors, audit.warnings


def compare_random_effects(
    ri_result: RandomInterceptFitResult,
    ri_rs_result: RandomEffectsFitResult,
    registry: RandomEffectsComparisonRegistry,
) -> RandomEffectsComparisonResult:
    """Compare paired RI and RI+RS records without selecting a structure."""

    ri_records = tuple(ri_result.fit_records)
    rs_records = tuple(ri_rs_result.fit_records)
    ri_by_key = _records_by_key(ri_records)
    rs_by_key = _records_by_key(rs_records)
    expected_keys = tuple(
        (comparison.condition_name, comparison.target_name)
        for comparison in registry.comparisons
    )

    errors: list[str] = list(ri_result.errors) + list(ri_rs_result.errors)
    warnings: list[str] = list(ri_result.warnings) + list(ri_rs_result.warnings)
    audits: list[RandomEffectsPairAudit] = []
    derived_by_record: dict[int, dict[str, Any]] = {}
    expected_record_ids: set[int] = set()

    for condition_name, target_name in expected_keys:
        audit, ri_record, rs_record, pair_errors, pair_warnings = _pair_audit(
            condition_name,
            target_name,
            ri_by_key.get((condition_name, target_name), []),
            rs_by_key.get((condition_name, target_name), []),
        )
        audits.append(audit)
        errors.extend(pair_errors)
        warnings.extend(pair_warnings)

        if ri_record is None or rs_record is None or audit.status == "failed":
            continue

        expected_record_ids.update((id(ri_record), id(rs_record)))
        ri_delta_aic, rs_delta_aic, warning = _metric_deltas(
            ri_record,
            rs_record,
            "aic",
            "AIC",
            condition_name,
            target_name,
        )
        if warning:
            warnings.append(warning)
        ri_delta_bic, rs_delta_bic, warning = _metric_deltas(
            ri_record,
            rs_record,
            "bic",
            "BIC",
            condition_name,
            target_name,
        )
        if warning:
            warnings.append(warning)
        likelihood_ratio_statistic, warning = _likelihood_ratio(
            ri_record,
            rs_record,
            condition_name,
            target_name,
        )
        if warning:
            warnings.append(warning)

        derived_by_record[id(ri_record)] = {
            "delta_aic": ri_delta_aic,
            "delta_bic": ri_delta_bic,
            "likelihood_ratio_statistic": likelihood_ratio_statistic,
            "likelihood_ratio_p_value": None,
            "likelihood_ratio_role": LIKELIHOOD_RATIO_ROLE,
            "selection_status": SELECTION_STATUS,
        }
        derived_by_record[id(rs_record)] = {
            "delta_aic": rs_delta_aic,
            "delta_bic": rs_delta_bic,
            "likelihood_ratio_statistic": likelihood_ratio_statistic,
            "likelihood_ratio_p_value": None,
            "likelihood_ratio_role": LIKELIHOOD_RATIO_ROLE,
            "selection_status": SELECTION_STATUS,
        }

    expected_key_set = set(expected_keys)
    for source_name, grouped in (("RI", ri_by_key), ("RI_RS", rs_by_key)):
        for key, records in grouped.items():
            if key not in expected_key_set:
                errors.append(
                    f"unexpected {source_name} record for "
                    f"{key[0]}/{key[1]}"
                )
                audits.append(
                    RandomEffectsPairAudit(
                        condition_name=key[0],
                        target_name=key[1],
                        ri_record_count=len(records) if source_name == "RI" else 0,
                        ri_rs_record_count=len(records) if source_name == "RI_RS" else 0,
                        ri_included_row_signature=(
                            getattr(records[0], "included_row_signature", None)
                            if source_name == "RI"
                            else None
                        ),
                        ri_rs_included_row_signature=(
                            getattr(records[0], "included_row_signature", None)
                            if source_name == "RI_RS"
                            else None
                        ),
                        ri_observation_count=(
                            getattr(records[0], "observation_count", None)
                            if source_name == "RI"
                            else None
                        ),
                        ri_rs_observation_count=(
                            getattr(records[0], "observation_count", None)
                            if source_name == "RI_RS"
                            else None
                        ),
                        ri_participant_count=(
                            getattr(records[0], "participant_count", None)
                            if source_name == "RI"
                            else None
                        ),
                        ri_rs_participant_count=(
                            getattr(records[0], "participant_count", None)
                            if source_name == "RI_RS"
                            else None
                        ),
                        same_observations=False,
                        status="failed",
                        likelihood_ratio_statistic=None,
                        errors=(
                            f"unexpected {source_name} record for "
                            f"{key[0]}/{key[1]}",
                        ),
                        warnings=(),
                    )
                )

    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for key in expected_keys:
        for record in ri_by_key.get(key, []) + rs_by_key.get(key, []):
            seen_ids.add(id(record))
            row = _record_row(record, registry)
            row.update(derived_by_record.get(id(record), {
                "delta_aic": None,
                "delta_bic": None,
                "likelihood_ratio_statistic": None,
                "likelihood_ratio_p_value": None,
            }))
            rows.append(row)

    for record in ri_records + rs_records:
        if id(record) in seen_ids or id(record) in expected_record_ids:
            continue
        rows.append(_record_row(record, registry))

    comparison_columns = tuple(registry.comparison_fields) + COMPARISON_EXTRA_FIELDS
    comparison_table = pd.DataFrame(rows, columns=comparison_columns).copy(deep=True)
    frozen_audits = tuple(audits)
    return RandomEffectsComparisonResult(
        valid=not errors and all(audit.status == "comparable" for audit in frozen_audits),
        errors=_unique(errors),
        warnings=_unique(warnings),
        pair_audits=frozen_audits,
        comparison_table=comparison_table,
    )
