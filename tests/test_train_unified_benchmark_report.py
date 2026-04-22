import json
import subprocess
import sys
from pathlib import Path


def test_train_unified_benchmark_report_smoke(tmp_path):
    output_dir = tmp_path / "unified"
    summary_plot_path = tmp_path / "unified_summary.png"
    summary_metrics_path = tmp_path / "unified_summary.json"
    summary_markdown_path = tmp_path / "unified_summary.md"

    cmd = [
        sys.executable,
        "train_unified_benchmark_report.py",
        "--models",
        "classical,quantum_attention",
        "--qa_models",
        "quantum_attention",
        "--primary_score_mode",
        "fidelity",
        "--compare_score_modes",
        "fidelity",
        "--seed",
        "7",
        "--output_dir",
        str(output_dir),
        "--summary_plot_path",
        str(summary_plot_path),
        "--summary_metrics_path",
        str(summary_metrics_path),
        "--summary_markdown_path",
        str(summary_markdown_path),
        "--lm_dataset",
        "tiny",
        "--lm_epochs",
        "1",
        "--lm_train_steps",
        "4",
        "--lm_eval_steps",
        "2",
        "--lm_vocab_max",
        "100",
        "--lm_block",
        "4",
        "--classification_epochs",
        "1",
        "--classification_batch_size",
        "4",
        "--classification_train_size",
        "8",
        "--classification_val_size",
        "4",
        "--classification_seq_len",
        "6",
        "--classification_vocab_size",
        "20",
        "--classification_embed_dim",
        "8",
        "--classification_n_qubits",
        "3",
        "--classification_n_layers",
        "1",
        "--classification_shots",
        "10",
    ]

    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    assert output_dir.exists()
    assert summary_plot_path.exists()
    assert summary_metrics_path.exists()
    assert summary_markdown_path.exists()

    payload = json.loads(summary_metrics_path.read_text(encoding="utf-8"))
    assert "tasks" in payload
    assert "language_modeling" in payload["tasks"]
    assert "classification" in payload["tasks"]
    assert "score_mode_comparisons" in payload
