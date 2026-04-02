"""
Language modeling benchmark for classical and quantum-hybrid transformer variants.
"""

from typing import Callable, Dict, Iterable, List, Tuple
import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import PennTreebank, WikiText2
from torchtext.vocab import Vocab, build_vocab_from_iterator

from src.benchmarking import count_parameters, write_json
from src.transformer_lm import (
    ClassicalTransformerLM,
    QuantumAttentionTransformerLM,
    QuantumEmbeddingQuantumAttentionTransformerLM,
    QuantumEmbeddingTransformerLM,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_MAX_SIZE = 2000
MIN_FREQ = 2
BLOCK_SIZE = 32
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

EMBED_DIM = 16
NUM_HEADS = 1
MAX_SEQ_LEN = 256

N_QUBITS = 6
N_LAYERS = 2
SHOTS = 300

EPOCHS = 2
TRAIN_STEPS_LIMIT = 400
EVAL_STEPS_LIMIT = 100
LR = 3e-4


def yield_tokens(data_iter: Iterable[str], tokenizer) -> Iterable[List[str]]:
    for text in data_iter:
        yield tokenizer(text)


def encode_corpus(split_iter: Iterable[str], vocab: Vocab, tokenizer) -> torch.Tensor:
    token_ids: List[int] = []
    for line in split_iter:
        token_ids.extend(vocab(tokenizer(line)))
    return torch.tensor(token_ids, dtype=torch.long)


class LMSequenceDataset(Dataset):
    """Fixed-length windows for next-token prediction."""

    def __init__(self, token_ids: torch.Tensor, block_size: int) -> None:
        super().__init__()
        self.tokens = token_ids
        self.block = block_size
        self.n = (len(token_ids) - 1) // block_size

    def __len__(self) -> int:
        return max(0, self.n)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block
        x = self.tokens[start : start + self.block]
        y = self.tokens[start + 1 : start + self.block + 1]
        return x, y


def batchify(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion) -> torch.Tensor:
    logits = model(x)
    batch_size, seq_len, vocab_size = logits.shape
    return criterion(logits.reshape(batch_size * seq_len, vocab_size), y.reshape(batch_size * seq_len))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, steps_limit: int = None) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    steps = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        loss = compute_loss(model, x, y, criterion)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
        steps += 1
        if steps_limit and steps >= steps_limit:
            break
    if total_tokens == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / total_tokens
    return avg_loss, float(torch.exp(torch.tensor(avg_loss)))


def train_one_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
) -> dict:
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "epoch_time_sec": [],
        "avg_train_step_ms": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        steps = 0
        train_step_time = 0.0
        epoch_start = time.perf_counter()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            step_start = time.perf_counter()
            optimizer.zero_grad()
            loss = compute_loss(model, x, y, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_step_time += time.perf_counter() - step_start
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            steps += 1
            if TRAIN_STEPS_LIMIT and steps >= TRAIN_STEPS_LIMIT:
                break

        avg_train = total_loss / max(1, total_tokens)
        train_ppl = float(torch.exp(torch.tensor(avg_train)))
        val_loss, val_ppl = evaluate(model, val_loader, criterion, steps_limit=EVAL_STEPS_LIMIT)
        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(avg_train)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["epoch_time_sec"].append(epoch_time)
        history["avg_train_step_ms"].append((train_step_time / max(1, steps)) * 1000.0)

        print(
            f"{model_name} | Epoch {ep}/{epochs} | "
            f"Train loss {avg_train:.4f} | Train ppl {train_ppl:.2f} | "
            f"Val loss {val_loss:.4f} | Val ppl {val_ppl:.2f} | "
            f"Epoch {epoch_time:.2f}s"
        )

    return {
        "history": history,
        "model": model,
        "parameter_counts": count_parameters(model),
    }


@torch.no_grad()
def generate(model: nn.Module, vocab: Vocab, prompt: str, block_size: int, max_new_tokens: int = 40) -> str:
    model.eval()
    tokenizer = get_tokenizer("basic_english")
    ids = torch.tensor([vocab[token] for token in tokenizer(prompt)], dtype=torch.long, device=DEVICE)
    if len(ids) == 0:
        ids = torch.tensor([vocab["<unk>"]], dtype=torch.long, device=DEVICE)
    ids = ids[-block_size:].unsqueeze(0)

    for _ in range(max_new_tokens):
        x = ids[:, -block_size:]
        logits = model(x)
        next_id = int(torch.argmax(logits[0, -1]))
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)

    tokens = [vocab.get_itos()[idx] for idx in ids[0].tolist()]
    return " ".join(tokens)


def load_iters(which: str):
    if which == "ptb":
        return PennTreebank(split="train"), PennTreebank(split="valid")
    if which == "wikitext2":
        return WikiText2(split="train"), WikiText2(split="valid")
    if which == "tiny":
        tiny_train = [
            "the quick brown fox jumps over the lazy dog",
            "the meaning of life is not clear",
            "we test a tiny dataset for language modeling",
            "this is a small sample of text",
            "quantum embeddings are slow to simulate",
        ]
        tiny_valid = [
            "the quick brown fox",
            "language modeling on tiny data",
        ]
        return tiny_train, tiny_valid
    raise ValueError(which)


def build_model_factories(
    vocab_size: int,
    score_mode: str,
) -> Dict[str, Tuple[str, Callable[[], nn.Module]]]:
    return {
        "classical": (
            "ClassicalTransformerLM",
            lambda: ClassicalTransformerLM(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                max_seq_len=MAX_SEQ_LEN,
                num_heads=NUM_HEADS,
            ),
        ),
        "quantum_embedding": (
            "QuantumEmbeddingTransformerLM",
            lambda: QuantumEmbeddingTransformerLM(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                max_seq_len=MAX_SEQ_LEN,
                num_heads=NUM_HEADS,
                n_qubits=N_QUBITS,
                n_layers=N_LAYERS,
                shots=SHOTS,
            ),
        ),
        "quantum_attention": (
            "QuantumAttentionTransformerLM",
            lambda: QuantumAttentionTransformerLM(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                max_seq_len=MAX_SEQ_LEN,
                num_heads=NUM_HEADS,
                n_qubits=N_QUBITS,
                shots=SHOTS,
                score_mode=score_mode,
            ),
        ),
        "full_quantum": (
            "QuantumEmbeddingQuantumAttentionTransformerLM",
            lambda: QuantumEmbeddingQuantumAttentionTransformerLM(
                vocab_size=vocab_size,
                embed_dim=EMBED_DIM,
                max_seq_len=MAX_SEQ_LEN,
                num_heads=NUM_HEADS,
                n_qubits=N_QUBITS,
                n_layers=N_LAYERS,
                shots=SHOTS,
                score_mode=score_mode,
            ),
        ),
    }


def plot_histories(results: Dict[str, dict], plot_path: str) -> None:
    epochs = list(range(1, len(next(iter(results.values()))["history"]["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, result in results.items():
        history = result["history"]
        axes[0].plot(epochs, history["train_loss"], marker="o", label=f"{label} train")
        axes[0].plot(epochs, history["val_loss"], marker="o", linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, history["train_ppl"], marker="o", label=f"{label} train")
        axes[1].plot(epochs, history["val_ppl"], marker="o", linestyle="--", label=f"{label} val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Perplexity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PPL")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    plt.savefig(plot_path, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")


def main():
    global EPOCHS, TRAIN_STEPS_LIMIT, EVAL_STEPS_LIMIT, VOCAB_MAX_SIZE, BLOCK_SIZE

    parser = argparse.ArgumentParser(description="LM benchmark for classical and quantum-hybrid transformers")
    parser.add_argument("--dataset", choices=["ptb", "wikitext2", "tiny"], default="ptb")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train_steps", type=int, default=TRAIN_STEPS_LIMIT)
    parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS_LIMIT)
    parser.add_argument("--vocab_max", type=int, default=VOCAB_MAX_SIZE)
    parser.add_argument("--block", type=int, default=BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_path", type=str, default="lm_generation_comparison.png")
    parser.add_argument("--metrics_path", type=str, default="artifacts/lm_benchmark_metrics.json")
    parser.add_argument(
        "--models",
        type=str,
        default="classical,quantum_embedding,quantum_attention,full_quantum",
        help="Comma-separated model keys: classical, quantum_embedding, quantum_attention, full_quantum",
    )
    parser.add_argument(
        "--score_mode",
        choices=["swap_test", "fidelity"],
        default="fidelity",
        help="Scoring backend for quantum attention models.",
    )
    args = parser.parse_args()

    EPOCHS = args.epochs
    TRAIN_STEPS_LIMIT = args.train_steps
    EVAL_STEPS_LIMIT = args.eval_steps
    VOCAB_MAX_SIZE = args.vocab_max
    BLOCK_SIZE = args.block
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Using device: {DEVICE}")
    print(f"Dataset: {args.dataset}")

    train_iter, valid_iter = load_iters(args.dataset)
    tokenizer = get_tokenizer("basic_english")

    try:
        vocab = build_vocab_from_iterator(
            yield_tokens(train_iter, tokenizer) if args.dataset != "tiny" else (tokenizer(text) for text in train_iter),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_freq=MIN_FREQ,
            max_tokens=VOCAB_MAX_SIZE,
        )
    except Exception as exc:
        print(f"Falling back to tiny offline dataset due to error: {exc}")
        args.dataset = "tiny"
        train_iter, valid_iter = load_iters("tiny")
        vocab = build_vocab_from_iterator(
            (tokenizer(text) for text in train_iter),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_freq=1,
            max_tokens=VOCAB_MAX_SIZE,
        )

    vocab.set_default_index(vocab["<unk>"])

    if args.dataset == "ptb":
        train_iter = PennTreebank(split="train")
        valid_iter = PennTreebank(split="valid")
    elif args.dataset == "wikitext2":
        train_iter = WikiText2(split="train")
        valid_iter = WikiText2(split="valid")

    train_ids = encode_corpus(train_iter, vocab, tokenizer)
    valid_ids = encode_corpus(valid_iter, vocab, tokenizer)
    print(f"Vocab size: {len(vocab)} | Train tokens: {len(train_ids)} | Valid tokens: {len(valid_ids)}")

    train_ds = LMSequenceDataset(train_ids, BLOCK_SIZE)
    valid_ds = LMSequenceDataset(valid_ids, BLOCK_SIZE)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=batchify)
    valid_loader = DataLoader(valid_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=batchify)

    factories = build_model_factories(len(vocab), args.score_mode)
    selected_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    unknown_keys = [key for key in selected_keys if key not in factories]
    if unknown_keys:
        raise ValueError(f"Unknown model keys: {', '.join(unknown_keys)}")

    results: Dict[str, dict] = {}
    generations: Dict[str, str] = {}
    prompt = "the meaning of life"

    for key in selected_keys:
        label, factory = factories[key]
        print(f"\n=== Training {label} ===")
        run_result = train_one_model(label, factory(), train_loader, valid_loader, EPOCHS)
        results[label] = run_result
        generations[label] = generate(run_result["model"], vocab, prompt, BLOCK_SIZE, max_new_tokens=30)

    print("\n=== Summary ===")
    for label, result in results.items():
        history = result["history"]
        print(
            f"{label} | Final train loss {history['train_loss'][-1]:.4f} | "
            f"Final val loss {history['val_loss'][-1]:.4f} | "
            f"Final val ppl {history['val_ppl'][-1]:.2f} | "
            f"Epoch time {history['epoch_time_sec'][-1]:.2f}s"
        )

    plot_histories(results, args.plot_path)

    for label, text in generations.items():
        print(f"\n--- Generation ({label}) ---")
        print(text)

    payload = {
        "config": {
            "dataset": args.dataset,
            "epochs": EPOCHS,
            "train_steps": TRAIN_STEPS_LIMIT,
            "eval_steps": EVAL_STEPS_LIMIT,
            "vocab_max": VOCAB_MAX_SIZE,
            "block_size": BLOCK_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "max_seq_len": MAX_SEQ_LEN,
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "shots": SHOTS,
            "seed": args.seed,
            "device": str(DEVICE),
            "score_mode": args.score_mode,
            "models": selected_keys,
            "plot_path": args.plot_path,
        },
        "results": {
            label: {
                "history": result["history"],
                "parameter_counts": result["parameter_counts"],
                "generation": generations[label],
            }
            for label, result in results.items()
        },
    }
    metrics_path = write_json(args.metrics_path, payload)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
