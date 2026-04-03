"""
Compare classical, quantum-embedding, quantum-attention, and full-hybrid
transformer classifiers on a synthetic sentiment task.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.benchmarking import count_parameters, write_json
from src.classical_transformer import ClassicalTransformer
from src.hybrid_quantum_embedding_transformer import HybridQuantumEmbeddingTransformer
from src.quantum_transformer_classifiers import (
    QuantumAttentionTransformer,
    QuantumEmbeddingQuantumAttentionTransformer,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EMBED_DIM = 8
NUM_CLASSES = 2
N_EPOCHS = 10
SEQ_LEN = 6
VOCAB_SIZE = 20
N_QUBITS = 3
N_LAYERS = 2
SHOTS = 300
TRAIN_SIZE = 800
VAL_SIZE = 200
LR = 1e-3


class SyntheticSentimentDataset(Dataset):
    """Small deterministic sentiment dataset for benchmark comparisons."""

    def __init__(self, size: int, seq_len: int, vocab_size: int, seed: int) -> None:
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.size):
            label = self.rng.randint(0, 1)
            if label == 1:
                sequence = [self.rng.randint(10, self.vocab_size - 1) for _ in range(self.seq_len // 2)]
                sequence += [self.rng.randint(1, self.vocab_size - 1) for _ in range(self.seq_len - len(sequence))]
            else:
                sequence = [self.rng.randint(1, 9) for _ in range(self.seq_len // 2)]
                sequence += [self.rng.randint(1, self.vocab_size - 1) for _ in range(self.seq_len - len(sequence))]
            self.rng.shuffle(sequence)
            if len(sequence) < self.seq_len:
                sequence += [0] * (self.seq_len - len(sequence))
            data.append((torch.tensor(sequence, dtype=torch.long), label))
        return data

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return self.data[idx]


def make_loaders(seed: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = SyntheticSentimentDataset(TRAIN_SIZE, SEQ_LEN, VOCAB_SIZE, seed=seed)
    val_dataset = SyntheticSentimentDataset(VAL_SIZE, SEQ_LEN, VOCAB_SIZE, seed=seed + 1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def build_factories(score_mode: str) -> Dict[str, Tuple[str, Callable[[], nn.Module]]]:
    return {
        "classical": (
            "ClassicalTransformer",
            lambda: ClassicalTransformer(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES),
        ),
        "quantum_embedding": (
            "HybridQuantumEmbeddingTransformer",
            lambda: HybridQuantumEmbeddingTransformer(
                VOCAB_SIZE,
                EMBED_DIM,
                NUM_CLASSES,
                n_qubits=N_QUBITS,
                n_layers=N_LAYERS,
                shots=SHOTS,
            ),
        ),
        "quantum_attention": (
            "QuantumAttentionTransformer",
            lambda: QuantumAttentionTransformer(
                VOCAB_SIZE,
                EMBED_DIM,
                NUM_CLASSES,
                n_qubits=N_QUBITS,
                shots=SHOTS,
                score_mode=score_mode,
            ),
        ),
        "full_quantum": (
            "QuantumEmbeddingQuantumAttentionTransformer",
            lambda: QuantumEmbeddingQuantumAttentionTransformer(
                VOCAB_SIZE,
                EMBED_DIM,
                NUM_CLASSES,
                n_qubits=N_QUBITS,
                n_layers=N_LAYERS,
                shots=SHOTS,
                score_mode=score_mode,
            ),
        ),
    }


def plot_results(results: Dict[str, dict], plot_path: str) -> None:
    epochs = range(1, len(next(iter(results.values()))["train_losses"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, result in results.items():
        axes[0].plot(epochs, result["train_losses"], marker="o", label=f"{label} train")
        axes[0].plot(epochs, result["val_losses"], marker="o", linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, result["train_accs"], marker="o", label=f"{label} train")
        axes[1].plot(epochs, result["val_accs"], marker="o", linestyle="--", label=f"{label} val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    plt.savefig(plot_path, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")


def run_experiment(model_name: str, factory: Callable[[], nn.Module], train_loader, val_loader, epochs: int):
    print(f"\n=== Training {model_name} ===")
    model = factory().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    total_params = count_parameters(model)
    print(
        f"Parameters: total={total_params['total_parameters']:,} "
        f"trainable={total_params['trainable_parameters']:,}"
    )
    if hasattr(model, "get_quantum_info"):
        info = model.get_quantum_info()
        print(f"Quantum info: {info}")

    test_x = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN)).to(DEVICE)
    with torch.no_grad():
        test_out = model(test_x)
        print(f"Forward pass OK: {tuple(test_out.shape)}")

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    epoch_times: List[float] = []

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        epoch_times.append(time.perf_counter() - t0)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={epoch_times[-1]:.2f}s"
        )

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "epoch_times": epoch_times,
        "parameter_counts": total_params,
    }


def main():
    global BATCH_SIZE, EMBED_DIM, NUM_CLASSES, N_EPOCHS, SEQ_LEN, VOCAB_SIZE, N_QUBITS, N_LAYERS, SHOTS, TRAIN_SIZE, VAL_SIZE

    parser = argparse.ArgumentParser(description="Classical vs quantum attention classifier benchmark")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val_size", type=int, default=VAL_SIZE)
    parser.add_argument("--n_qubits", type=int, default=N_QUBITS)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_path", type=str, default="quantum_attention_classification_comparison.png")
    parser.add_argument("--metrics_path", type=str, default="artifacts/classification_benchmark_metrics.json")
    parser.add_argument(
        "--models",
        type=str,
        default="classical,quantum_embedding,quantum_attention,full_quantum",
        help="Comma-separated model keys to run",
    )
    parser.add_argument(
        "--score_mode",
        choices=["swap_test", "fidelity"],
        default="fidelity",
        help="Quantum attention scoring backend",
    )
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    EMBED_DIM = args.embed_dim
    SEQ_LEN = args.seq_len
    VOCAB_SIZE = args.vocab_size
    TRAIN_SIZE = args.train_size
    VAL_SIZE = args.val_size
    N_QUBITS = args.n_qubits
    N_LAYERS = args.n_layers
    SHOTS = args.shots
    N_EPOCHS = args.epochs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Using device: {DEVICE}")
    train_loader, val_loader = make_loaders(args.seed)
    factories = build_factories(args.score_mode)
    selected = [key.strip() for key in args.models.split(",") if key.strip()]
    unknown = [key for key in selected if key not in factories]
    if unknown:
        raise ValueError(f"Unknown model keys: {', '.join(unknown)}")

    results: Dict[str, dict] = {}
    for key in selected:
        label, factory = factories[key]
        results[label] = run_experiment(label, factory, train_loader, val_loader, N_EPOCHS)

    print("\n=== Summary ===")
    for label, result in results.items():
        print(
            f"{label} | final_train_loss={result['train_losses'][-1]:.4f} "
            f"final_val_loss={result['val_losses'][-1]:.4f} "
            f"final_val_acc={result['val_accs'][-1]:.4f}"
        )

    plot_results(results, args.plot_path)

    payload = {
        "config": {
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "seq_len": SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "train_size": TRAIN_SIZE,
            "val_size": VAL_SIZE,
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "shots": SHOTS,
            "seed": args.seed,
            "device": str(DEVICE),
            "score_mode": args.score_mode,
            "models": selected,
            "plot_path": args.plot_path,
        },
        "results": {
            label: {
                "train_losses": result["train_losses"],
                "val_losses": result["val_losses"],
                "train_accs": result["train_accs"],
                "val_accs": result["val_accs"],
                "epoch_times": result["epoch_times"],
                "parameter_counts": result["parameter_counts"],
            }
            for label, result in results.items()
        },
    }
    metrics_path = write_json(args.metrics_path, payload)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
