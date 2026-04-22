"""Utilities for reproducible benchmark reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import matplotlib.pyplot as plt
import torch.nn as nn


VARIANT_ORDER = [
    "classical",
    "quantum_embedding",
    "quantum_attention",
    "full_quantum",
]

VARIANT_DISPLAY_NAMES = {
    "classical": "Classical",
    "quantum_embedding": "Quantum Embedding",
    "quantum_attention": "Quantum Attention",
    "full_quantum": "Full Hybrid",
}

MODEL_LABEL_TO_VARIANT = {
    "ClassicalTransformerLM": "classical",
    "QuantumEmbeddingTransformerLM": "quantum_embedding",
    "QuantumAttentionTransformerLM": "quantum_attention",
    "QuantumEmbeddingQuantumAttentionTransformerLM": "full_quantum",
    "ClassicalTransformer": "classical",
    "HybridQuantumEmbeddingTransformer": "quantum_embedding",
    "QuantumAttentionTransformer": "quantum_attention",
    "QuantumEmbeddingQuantumAttentionTransformer": "full_quantum",
}

TASK_SPECS = {
    "language_modeling": {
        "primary_metric_key": "final_val_ppl",
        "primary_metric_label": "Final Val PPL",
        "secondary_metric_key": "final_val_loss",
        "secondary_metric_label": "Final Val Loss",
        "optimize": "min",
    },
    "classification": {
        "primary_metric_key": "final_val_acc",
        "primary_metric_label": "Final Val Acc",
        "secondary_metric_key": "final_val_loss",
        "secondary_metric_label": "Final Val Loss",
        "optimize": "max",
    },
}


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def write_json(path: str, payload: Dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _variant_key_for_label(label: str) -> str:
    try:
        return MODEL_LABEL_TO_VARIANT[label]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark model label: {label}") from exc


def _sort_rows(rows: Iterable[MutableMapping[str, Any]]) -> List[MutableMapping[str, Any]]:
    return sorted(rows, key=lambda row: VARIANT_ORDER.index(row["variant_key"]))


def _safe_pct(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return (numerator / denominator) * 100.0


def _attach_baseline_deltas(rows: List[MutableMapping[str, Any]], task_name: str) -> None:
    spec = TASK_SPECS[task_name]
    baseline = next((row for row in rows if row["variant_key"] == "classical"), None)
    if baseline is None:
        return

    primary_key = spec["primary_metric_key"]
    baseline_value = baseline[primary_key]
    optimize = spec["optimize"]
    for row in rows:
        raw_delta = row[primary_key] - baseline_value
        if optimize == "min":
            improvement = baseline_value - row[primary_key]
        else:
            improvement = row[primary_key] - baseline_value
        row["delta_vs_classical"] = raw_delta
        row["improvement_vs_classical_pct"] = _safe_pct(improvement, baseline_value)


def summarize_lm_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows: List[MutableMapping[str, Any]] = []
    for model_label, result in payload["results"].items():
        history = result["history"]
        row: MutableMapping[str, Any] = {
            "variant_key": _variant_key_for_label(model_label),
            "model_label": model_label,
            "display_name": VARIANT_DISPLAY_NAMES[_variant_key_for_label(model_label)],
            "final_train_loss": history["train_loss"][-1],
            "final_train_ppl": history["train_ppl"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_ppl": history["val_ppl"][-1],
            "best_val_loss": min(history["val_loss"]),
            "best_val_ppl": min(history["val_ppl"]),
            "final_epoch_time_sec": history["epoch_time_sec"][-1],
            "avg_epoch_time_sec": sum(history["epoch_time_sec"]) / max(1, len(history["epoch_time_sec"])),
            "avg_train_step_ms": sum(history["avg_train_step_ms"]) / max(1, len(history["avg_train_step_ms"])),
            "parameter_counts": result["parameter_counts"],
            "generation": result.get("generation"),
        }
        rows.append(row)

    rows = _sort_rows(rows)
    _attach_baseline_deltas(rows, "language_modeling")
    primary_key = TASK_SPECS["language_modeling"]["primary_metric_key"]
    winner = min(rows, key=lambda row: row[primary_key]) if rows else None
    return {
        "task": "language_modeling",
        **TASK_SPECS["language_modeling"],
        "config": dict(payload["config"]),
        "rows": rows,
        "winner_variant": None if winner is None else winner["variant_key"],
        "winner_model_label": None if winner is None else winner["model_label"],
    }


def summarize_classification_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows: List[MutableMapping[str, Any]] = []
    for model_label, result in payload["results"].items():
        row: MutableMapping[str, Any] = {
            "variant_key": _variant_key_for_label(model_label),
            "model_label": model_label,
            "display_name": VARIANT_DISPLAY_NAMES[_variant_key_for_label(model_label)],
            "final_train_loss": result["train_losses"][-1],
            "final_val_loss": result["val_losses"][-1],
            "final_train_acc": result["train_accs"][-1],
            "final_val_acc": result["val_accs"][-1],
            "best_val_loss": min(result["val_losses"]),
            "best_val_acc": max(result["val_accs"]),
            "final_epoch_time_sec": result["epoch_times"][-1],
            "avg_epoch_time_sec": sum(result["epoch_times"]) / max(1, len(result["epoch_times"])),
            "parameter_counts": result["parameter_counts"],
        }
        rows.append(row)

    rows = _sort_rows(rows)
    _attach_baseline_deltas(rows, "classification")
    primary_key = TASK_SPECS["classification"]["primary_metric_key"]
    winner = max(rows, key=lambda row: row[primary_key]) if rows else None
    return {
        "task": "classification",
        **TASK_SPECS["classification"],
        "config": dict(payload["config"]),
        "rows": rows,
        "winner_variant": None if winner is None else winner["variant_key"],
        "winner_model_label": None if winner is None else winner["model_label"],
    }


def _shared_variants(mode_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    variant_sets = [
        {row["variant_key"] for row in rows}
        for rows in mode_rows.values()
        if rows
    ]
    if not variant_sets:
        return []
    shared = set.intersection(*variant_sets)
    return [variant for variant in VARIANT_ORDER if variant in shared]


def build_score_mode_comparison(
    task_name: str,
    summaries_by_mode: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    if not summaries_by_mode:
        return {
            "task": task_name,
            **TASK_SPECS[task_name],
            "score_modes": [],
            "rows": [],
        }

    spec = TASK_SPECS[task_name]
    score_modes = list(summaries_by_mode.keys())
    mode_rows = {mode: summary["rows"] for mode, summary in summaries_by_mode.items()}
    primary_key = spec["primary_metric_key"]
    shared_variants = _shared_variants(mode_rows)
    reference_mode = "fidelity" if "fidelity" in summaries_by_mode else score_modes[0]
    runtime_key = "avg_epoch_time_sec"
    runtime_label = "Avg Epoch (s)"
    if all(
        all("eval_time_sec" in row for row in summary["rows"])
        for summary in summaries_by_mode.values()
    ):
        runtime_key = "eval_time_sec"
        runtime_label = "Eval Time (s)"

    rows = []
    for variant_key in shared_variants:
        per_mode = {}
        for mode, summary in summaries_by_mode.items():
            row = next(item for item in summary["rows"] if item["variant_key"] == variant_key)
            per_mode[mode] = {
                primary_key: row[primary_key],
                "runtime_sec": row[runtime_key],
            }

        comparison_row: Dict[str, Any] = {
            "variant_key": variant_key,
            "display_name": VARIANT_DISPLAY_NAMES[variant_key],
            "per_mode": per_mode,
        }
        if reference_mode in per_mode:
            reference = per_mode[reference_mode]
            deltas = {}
            for mode, values in per_mode.items():
                raw_delta = values[primary_key] - reference[primary_key]
                if spec["optimize"] == "min":
                    improvement = reference[primary_key] - values[primary_key]
                else:
                    improvement = values[primary_key] - reference[primary_key]
                runtime_ratio = (
                    values["runtime_sec"] / reference["runtime_sec"]
                    if reference["runtime_sec"] > 0
                    else 0.0
                )
                deltas[mode] = {
                    "primary_metric_delta_vs_reference": raw_delta,
                    "primary_metric_improvement_pct_vs_reference": _safe_pct(improvement, reference[primary_key]),
                    "avg_epoch_time_ratio_vs_reference": runtime_ratio,
                }
            comparison_row["reference_mode"] = reference_mode
            comparison_row["deltas_vs_reference"] = deltas
        rows.append(comparison_row)

    return {
        "task": task_name,
        **spec,
        "runtime_key": runtime_key,
        "runtime_label": runtime_label,
        "score_modes": score_modes,
        "rows": rows,
    }


def build_unified_benchmark_summary(
    lm_payload: Mapping[str, Any],
    classification_payload: Mapping[str, Any],
    score_mode_payloads: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    lm_summary = summarize_lm_metrics(lm_payload)
    classification_summary = summarize_classification_metrics(classification_payload)

    score_mode_payloads = score_mode_payloads or {}
    score_mode_summaries: Dict[str, Any] = {}
    if "language_modeling" in score_mode_payloads:
        lm_modes = {
            mode: summarize_lm_metrics(payload)
            for mode, payload in score_mode_payloads["language_modeling"].items()
        }
        score_mode_summaries["language_modeling"] = build_score_mode_comparison("language_modeling", lm_modes)
    if "classification" in score_mode_payloads:
        cls_modes = {
            mode: summarize_classification_metrics(payload)
            for mode, payload in score_mode_payloads["classification"].items()
        }
        score_mode_summaries["classification"] = build_score_mode_comparison("classification", cls_modes)

    evidence = {}
    for variant_key in ("quantum_attention", "full_quantum"):
        evidence[variant_key] = {}
        for task_name, summary in (
            ("language_modeling", lm_summary),
            ("classification", classification_summary),
        ):
            row = next((item for item in summary["rows"] if item["variant_key"] == variant_key), None)
            if row is None:
                continue
            evidence[variant_key][task_name] = {
                summary["primary_metric_key"]: row[summary["primary_metric_key"]],
                "delta_vs_classical": row.get("delta_vs_classical"),
                "improvement_vs_classical_pct": row.get("improvement_vs_classical_pct"),
            }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant_order": VARIANT_ORDER,
        "metadata": dict(metadata or {}),
        "tasks": {
            "language_modeling": lm_summary,
            "classification": classification_summary,
        },
        "score_mode_comparisons": score_mode_summaries,
        "quantum_attention_evidence": evidence,
    }
    return summary


def _format_value(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_unified_benchmark_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Unified Benchmark Summary", ""]
    metadata = summary.get("metadata", {})
    if metadata:
        lines.append("## Run Metadata")
        for key, value in metadata.items():
            lines.append(f"- **{key}**: `{value}`")
        lines.append("")

    for task_name in ("language_modeling", "classification"):
        task = summary["tasks"][task_name]
        lines.append(f"## {task_name.replace('_', ' ').title()}")
        lines.append(
            f"Primary metric: **{task['primary_metric_label']}** "
            f"({'lower is better' if task['optimize'] == 'min' else 'higher is better'})"
        )
        lines.append("")
        headers = [
            "Variant",
            task["primary_metric_label"],
            task["secondary_metric_label"],
            "Avg Epoch (s)",
            "Params",
            "Delta vs Classical",
        ]
        rows = []
        for row in task["rows"]:
            rows.append(
                [
                    row["display_name"],
                    _format_value(row[task["primary_metric_key"]], 4 if task_name == "classification" else 2),
                    _format_value(row[task["secondary_metric_key"]]),
                    _format_value(row["avg_epoch_time_sec"], 2),
                    f"{row['parameter_counts']['total_parameters']:,}",
                    _format_value(row.get("delta_vs_classical", 0.0), 4 if task_name == "classification" else 2),
                ]
            )
        lines.append(_markdown_table(headers, rows))
        lines.append("")
        winner_name = VARIANT_DISPLAY_NAMES.get(task["winner_variant"], task["winner_variant"])
        lines.append(f"Winner: **{winner_name}**")
        lines.append("")

    comparisons = summary.get("score_mode_comparisons", {})
    for task_name, comparison in comparisons.items():
        if not comparison["rows"]:
            continue
        lines.append(f"## {task_name.replace('_', ' ').title()} Score-Mode Comparison")
        headers = [
            "Variant",
            "Fidelity",
            "Swap Test",
            f"Fidelity {comparison['runtime_label']}",
            f"Swap Test {comparison['runtime_label']}",
            "Swap Test / Fidelity Time",
        ]
        rows = []
        metric_key = comparison["primary_metric_key"]
        for row in comparison["rows"]:
            fidelity = row["per_mode"].get("fidelity")
            swap = row["per_mode"].get("swap_test")
            if fidelity is None or swap is None:
                continue
            runtime_ratio = row["deltas_vs_reference"]["swap_test"]["avg_epoch_time_ratio_vs_reference"]
            digits = 4 if task_name == "classification" else 2
            rows.append(
                [
                    row["display_name"],
                    _format_value(fidelity[metric_key], digits),
                    _format_value(swap[metric_key], digits),
                    _format_value(fidelity["runtime_sec"], 2),
                    _format_value(swap["runtime_sec"], 2),
                    _format_value(runtime_ratio, 2),
                ]
            )
        if rows:
            lines.append(_markdown_table(headers, rows))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_markdown(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def plot_unified_benchmark_summary(summary: Mapping[str, Any], plot_path: str | Path) -> Path:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = {
        "classical": "#4C78A8",
        "quantum_embedding": "#F58518",
        "quantum_attention": "#54A24B",
        "full_quantum": "#E45756",
    }

    task_plot_specs = [
        ("language_modeling", axes[0, 0], "Final Val PPL"),
        ("classification", axes[0, 1], "Final Val Acc"),
    ]
    for task_name, axis, ylabel in task_plot_specs:
        task = summary["tasks"][task_name]
        variants = [row["display_name"] for row in task["rows"]]
        values = [row[task["primary_metric_key"]] for row in task["rows"]]
        bar_colors = [colors[row["variant_key"]] for row in task["rows"]]
        bars = axis.bar(variants, values, color=bar_colors)
        axis.set_title(task_name.replace("_", " ").title())
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        digits = 4 if task_name == "classification" else 2
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                _format_value(value, digits),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    comparison_specs = [
        ("language_modeling", axes[1, 0]),
        ("classification", axes[1, 1]),
    ]
    for task_name, axis in comparison_specs:
        comparison = summary.get("score_mode_comparisons", {}).get(task_name)
        if not comparison or not comparison["rows"]:
            axis.axis("off")
            continue

        metric_key = comparison["primary_metric_key"]
        variants = [
            row["display_name"]
            for row in comparison["rows"]
            if "fidelity" in row["per_mode"] and "swap_test" in row["per_mode"]
        ]
        fidelity_values = [
            row["per_mode"]["fidelity"][metric_key]
            for row in comparison["rows"]
            if "fidelity" in row["per_mode"] and "swap_test" in row["per_mode"]
        ]
        swap_values = [
            row["per_mode"]["swap_test"][metric_key]
            for row in comparison["rows"]
            if "fidelity" in row["per_mode"] and "swap_test" in row["per_mode"]
        ]
        if not variants:
            axis.axis("off")
            continue

        xs = range(len(variants))
        width = 0.35
        axis.bar([x - width / 2 for x in xs], fidelity_values, width=width, label="fidelity", color="#4C78A8")
        axis.bar([x + width / 2 for x in xs], swap_values, width=width, label="swap_test", color="#E45756")
        axis.set_xticks(list(xs))
        axis.set_xticklabels(variants, rotation=15)
        axis.set_title(f"{task_name.replace('_', ' ').title()} score modes")
        axis.set_ylabel(comparison["primary_metric_label"])
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path
