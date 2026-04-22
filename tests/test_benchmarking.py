import pytest

from src.benchmarking import (
    build_unified_benchmark_summary,
    render_unified_benchmark_markdown,
)


def _lm_payload(score_mode: str = "fidelity"):
    return {
        "config": {
            "dataset": "tiny",
            "score_mode": score_mode,
            "models": [
                "classical",
                "quantum_embedding",
                "quantum_attention",
                "full_quantum",
            ],
        },
        "results": {
            "ClassicalTransformerLM": {
                "history": {
                    "train_loss": [2.4, 2.2],
                    "train_ppl": [11.0, 9.0],
                    "val_loss": [2.3, 2.1],
                    "val_ppl": [10.0, 8.0],
                    "epoch_time_sec": [1.0, 1.2],
                    "avg_train_step_ms": [10.0, 12.0],
                },
                "parameter_counts": {"total_parameters": 100, "trainable_parameters": 100, "frozen_parameters": 0},
                "generation": "classical output",
            },
            "QuantumEmbeddingTransformerLM": {
                "history": {
                    "train_loss": [2.2, 2.0],
                    "train_ppl": [9.5, 7.5],
                    "val_loss": [2.1, 1.9],
                    "val_ppl": [8.5, 7.0],
                    "epoch_time_sec": [1.4, 1.5],
                    "avg_train_step_ms": [15.0, 16.0],
                },
                "parameter_counts": {"total_parameters": 120, "trainable_parameters": 120, "frozen_parameters": 0},
                "generation": "embedding output",
            },
            "QuantumAttentionTransformerLM": {
                "history": {
                    "train_loss": [2.5, 2.4],
                    "train_ppl": [12.0, 11.0],
                    "val_loss": [2.6, 2.5],
                    "val_ppl": [13.0, 12.0],
                    "epoch_time_sec": [2.0, 2.1],
                    "avg_train_step_ms": [21.0, 22.0],
                },
                "parameter_counts": {"total_parameters": 110, "trainable_parameters": 110, "frozen_parameters": 0},
                "generation": "attention output",
            },
            "QuantumEmbeddingQuantumAttentionTransformerLM": {
                "history": {
                    "train_loss": [2.3, 2.1],
                    "train_ppl": [10.5, 8.2],
                    "val_loss": [2.2, 2.0],
                    "val_ppl": [9.0, 7.8],
                    "epoch_time_sec": [2.4, 2.5],
                    "avg_train_step_ms": [24.0, 25.0],
                },
                "parameter_counts": {"total_parameters": 140, "trainable_parameters": 140, "frozen_parameters": 0},
                "generation": "full output",
            },
        },
    }


def _classification_payload(score_mode: str = "fidelity"):
    return {
        "config": {
            "score_mode": score_mode,
            "models": [
                "classical",
                "quantum_embedding",
                "quantum_attention",
                "full_quantum",
            ],
        },
        "results": {
            "ClassicalTransformer": {
                "train_losses": [0.7, 0.65],
                "val_losses": [0.72, 0.70],
                "train_accs": [0.50, 0.60],
                "val_accs": [0.50, 0.55],
                "epoch_times": [0.4, 0.5],
                "parameter_counts": {"total_parameters": 90, "trainable_parameters": 90, "frozen_parameters": 0},
            },
            "HybridQuantumEmbeddingTransformer": {
                "train_losses": [0.68, 0.60],
                "val_losses": [0.69, 0.62],
                "train_accs": [0.55, 0.70],
                "val_accs": [0.58, 0.68],
                "epoch_times": [0.7, 0.8],
                "parameter_counts": {"total_parameters": 100, "trainable_parameters": 100, "frozen_parameters": 0},
            },
            "QuantumAttentionTransformer": {
                "train_losses": [0.66, 0.59],
                "val_losses": [0.67, 0.60],
                "train_accs": [0.60, 0.72],
                "val_accs": [0.60, 0.75],
                "epoch_times": [0.9, 1.0],
                "parameter_counts": {"total_parameters": 95, "trainable_parameters": 95, "frozen_parameters": 0},
            },
            "QuantumEmbeddingQuantumAttentionTransformer": {
                "train_losses": [0.69, 0.61],
                "val_losses": [0.68, 0.61],
                "train_accs": [0.56, 0.69],
                "val_accs": [0.59, 0.70],
                "epoch_times": [1.1, 1.2],
                "parameter_counts": {"total_parameters": 110, "trainable_parameters": 110, "frozen_parameters": 0},
            },
        },
    }


def test_build_unified_benchmark_summary_tracks_cross_task_story():
    lm_payload = _lm_payload()
    classification_payload = _classification_payload()
    swap_lm_payload = _lm_payload(score_mode="swap_test")
    swap_lm_payload["results"]["QuantumAttentionTransformerLM"]["history"]["val_ppl"][-1] = 13.5
    swap_cls_payload = _classification_payload(score_mode="swap_test")
    swap_cls_payload["results"]["QuantumAttentionTransformer"]["val_accs"][-1] = 0.70

    summary = build_unified_benchmark_summary(
        lm_payload=lm_payload,
        classification_payload=classification_payload,
        score_mode_payloads={
            "language_modeling": {
                "fidelity": {
                    "config": lm_payload["config"],
                    "results": {
                        "QuantumAttentionTransformerLM": lm_payload["results"]["QuantumAttentionTransformerLM"],
                        "QuantumEmbeddingQuantumAttentionTransformerLM": lm_payload["results"]["QuantumEmbeddingQuantumAttentionTransformerLM"],
                    },
                },
                "swap_test": {
                    "config": swap_lm_payload["config"],
                    "results": {
                        "QuantumAttentionTransformerLM": swap_lm_payload["results"]["QuantumAttentionTransformerLM"],
                        "QuantumEmbeddingQuantumAttentionTransformerLM": swap_lm_payload["results"]["QuantumEmbeddingQuantumAttentionTransformerLM"],
                    },
                },
            },
            "classification": {
                "fidelity": {
                    "config": classification_payload["config"],
                    "results": {
                        "QuantumAttentionTransformer": classification_payload["results"]["QuantumAttentionTransformer"],
                        "QuantumEmbeddingQuantumAttentionTransformer": classification_payload["results"]["QuantumEmbeddingQuantumAttentionTransformer"],
                    },
                },
                "swap_test": {
                    "config": swap_cls_payload["config"],
                    "results": {
                        "QuantumAttentionTransformer": swap_cls_payload["results"]["QuantumAttentionTransformer"],
                        "QuantumEmbeddingQuantumAttentionTransformer": swap_cls_payload["results"]["QuantumEmbeddingQuantumAttentionTransformer"],
                    },
                },
            },
        },
        metadata={"seed": 42},
    )

    assert summary["tasks"]["language_modeling"]["winner_variant"] == "quantum_embedding"
    assert summary["tasks"]["classification"]["winner_variant"] == "quantum_attention"
    assert summary["quantum_attention_evidence"]["quantum_attention"]["language_modeling"]["delta_vs_classical"] == 4.0
    assert summary["quantum_attention_evidence"]["quantum_attention"]["classification"]["delta_vs_classical"] == pytest.approx(
        0.20
    )

    lm_mode_row = summary["score_mode_comparisons"]["language_modeling"]["rows"][0]
    assert lm_mode_row["variant_key"] == "quantum_attention"
    assert lm_mode_row["per_mode"]["swap_test"]["final_val_ppl"] == 13.5


def test_render_unified_benchmark_markdown_contains_primary_and_score_mode_tables():
    summary = build_unified_benchmark_summary(
        lm_payload=_lm_payload(),
        classification_payload=_classification_payload(),
        score_mode_payloads={},
        metadata={"seed": 42, "models": "classical,quantum_attention"},
    )

    markdown = render_unified_benchmark_markdown(summary)

    assert "# Unified Benchmark Summary" in markdown
    assert "Language Modeling" in markdown
    assert "Classification" in markdown
    assert "Quantum Attention" in markdown
    assert "Winner:" in markdown
