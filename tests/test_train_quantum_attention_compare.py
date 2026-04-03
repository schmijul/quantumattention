import subprocess
import sys
from pathlib import Path


def test_train_quantum_attention_compare_smoke(tmp_path):
    plot_path = tmp_path / "quantum_attention_classification_smoke.png"
    metrics_path = tmp_path / "quantum_attention_classification_smoke.json"

    cmd = [
        sys.executable,
        "train_quantum_attention_compare.py",
        "--epochs",
        "1",
        "--batch_size",
        "8",
        "--train_size",
        "16",
        "--val_size",
        "8",
        "--seq_len",
        "6",
        "--vocab_size",
        "20",
        "--embed_dim",
        "8",
        "--n_qubits",
        "3",
        "--n_layers",
        "1",
        "--shots",
        "20",
        "--models",
        "classical,quantum_attention",
        "--score_mode",
        "fidelity",
        "--plot_path",
        str(plot_path),
        "--metrics_path",
        str(metrics_path),
    ]

    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    assert plot_path.exists()
    assert metrics_path.exists()
