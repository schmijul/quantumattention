"""
Language Modeling: Classical vs Quantum-Embedding Transformer

This script downloads a real text dataset (WikiText-2 via torchtext),
builds a capped vocabulary, and compares a classical Transformer LM
against a hybrid model that uses a Quantum Embedding layer with
classical attention/FFN for next-token prediction.

It trains briefly to keep runtime reasonable and reports:
- Train/valid loss and perplexity
- A short greedy generation sample for each model

Notes
- Quantum simulation is expensive. We cap the vocabulary size and
  use short sequences and few steps for a quick comparison.
"""

from pathlib import Path
from typing import Iterable, List, Tuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import PennTreebank, WikiText2

from src.transformer_lm import ClassicalTransformerLM, QuantumEmbeddingTransformerLM


# -----------------------
# Config
# -----------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
VOCAB_MAX_SIZE = 2000   # cap vocab for quantum embedding size
MIN_FREQ = 2
BLOCK_SIZE = 32         # sequence length
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

# Model
EMBED_DIM = 16
NUM_HEADS = 1
MAX_SEQ_LEN = 256

# Quantum
N_QUBITS = 6
N_LAYERS = 2
SHOTS = 300

# Train
EPOCHS = 2
TRAIN_STEPS_LIMIT = 400   # limit steps per epoch for speed
EVAL_STEPS_LIMIT = 100
LR = 3e-4


# -----------------------
# Data utilities
# -----------------------

def yield_tokens(data_iter: Iterable[str], tokenizer) -> Iterable[List[str]]:
    for text in data_iter:
        yield tokenizer(text)


def encode_corpus(split_iter: Iterable[str], vocab: Vocab, tokenizer) -> torch.Tensor:
    ids: List[int] = []
    for line in split_iter:
        ids.extend(vocab(tokenizer(line)))
    return torch.tensor(ids, dtype=torch.long)


class LMSequenceDataset(Dataset):
    """Create fixed-length sequences for next-token prediction.

    Given a long tensor of token ids, returns pairs:
      x[i] = tokens[s:s+block]
      y[i] = tokens[s+1:s+block+1]
    with stride = block.
    """

    def __init__(self, token_ids: torch.Tensor, block_size: int) -> None:
        super().__init__()
        self.tokens = token_ids
        self.block = block_size
        # last index such that s+block < len(tokens)
        self.n = (len(token_ids) - 1) // block_size

    def __len__(self) -> int:
        return max(0, self.n)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.block
        x = self.tokens[s : s + self.block]
        y = self.tokens[s + 1 : s + self.block + 1]
        return x, y


def batchify(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs)
    y = torch.stack(ys)
    return x, y


# -----------------------
# Training utilities
# -----------------------

def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion) -> torch.Tensor:
    logits = model(x)
    # shift targets already done in dataset; compute CE per position
    B, T, V = logits.shape
    loss = criterion(logits.reshape(B * T, V), y.reshape(B * T))
    return loss


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
        return float('inf'), float('inf')
    avg_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    return avg_loss, ppl


def train_one_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> dict:
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_ppl": [], "val_loss": [], "val_ppl": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        steps = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            loss = compute_loss(model, x, y, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            steps += 1
            if TRAIN_STEPS_LIMIT and steps >= TRAIN_STEPS_LIMIT:
                break

        avg_train = total_loss / max(1, total_tokens)
        train_ppl = float(torch.exp(torch.tensor(avg_train)))
        val_loss, val_ppl = evaluate(model, val_loader, criterion, steps_limit=EVAL_STEPS_LIMIT)

        history["train_loss"].append(avg_train)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        print(
            f"Epoch {ep}/{epochs} | Train loss {avg_train:.4f} | Train ppl {train_ppl:.2f} | "
            f"Val loss {val_loss:.4f} | Val ppl {val_ppl:.2f}"
        )

    return {"history": history, "model": model}


@torch.no_grad()
def generate(model: nn.Module, vocab: Vocab, prompt: str, max_new_tokens: int = 40) -> str:
    model.eval()
    # Use basic_english tokenizer to map prompt -> ids
    tokenizer = get_tokenizer("basic_english")
    ids = torch.tensor([vocab[token] for token in tokenizer(prompt)], dtype=torch.long, device=DEVICE)
    if len(ids) == 0:
        ids = torch.tensor([vocab["<unk>"]], dtype=torch.long, device=DEVICE)
    ids = ids[-BLOCK_SIZE:]  # crop to block
    ids = ids.unsqueeze(0)   # (1, T)

    for _ in range(max_new_tokens):
        # take last BLOCK_SIZE tokens as context
        x = ids[:, -BLOCK_SIZE:]
        logits = model(x)
        next_token_logits = logits[0, -1]  # (V,)
        next_id = int(torch.argmax(next_token_logits))
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    # map ids back to tokens
    itos = vocab.get_itos()
    toks = [itos[i] for i in ids[0].tolist()]
    return " ".join(toks)


def main():
    global EPOCHS, TRAIN_STEPS_LIMIT, EVAL_STEPS_LIMIT, VOCAB_MAX_SIZE, BLOCK_SIZE
    parser = argparse.ArgumentParser(description="Classical vs Quantum-Embedding Transformer LM")
    parser.add_argument("--dataset", choices=["ptb", "wikitext2", "tiny"], default="ptb",
                        help="Which dataset to use. 'tiny' is an offline built-in sample.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train_steps", type=int, default=TRAIN_STEPS_LIMIT)
    parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS_LIMIT)
    parser.add_argument("--vocab_max", type=int, default=VOCAB_MAX_SIZE)
    parser.add_argument("--block", type=int, default=BLOCK_SIZE)
    args = parser.parse_args()

    # Apply overrides
    EPOCHS = args.epochs
    TRAIN_STEPS_LIMIT = args.train_steps
    EVAL_STEPS_LIMIT = args.eval_steps
    VOCAB_MAX_SIZE = args.vocab_max
    BLOCK_SIZE = args.block

    print(f"Using device: {DEVICE}")
    print(f"Dataset: {args.dataset}")

    # Load dataset (tiny is offline, others require network on first run)
    def load_iters(which: str):
        if which == "ptb":
            return PennTreebank(split='train'), PennTreebank(split='valid')
        if which == "wikitext2":
            return WikiText2(split='train'), WikiText2(split='valid')
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

    print("Building vocab (capped)...")
    train_iter, valid_iter = load_iters(args.dataset)

    tokenizer = get_tokenizer("basic_english")
    try:
        vocab = build_vocab_from_iterator(
            yield_tokens(train_iter, tokenizer) if args.dataset != "tiny" else (tokenizer(x) for x in train_iter),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_freq=MIN_FREQ,
            max_tokens=VOCAB_MAX_SIZE,
        )
    except Exception as e:
        print(f"Falling back to tiny offline dataset due to error: {e}")
        args.dataset = "tiny"
        train_iter, valid_iter = load_iters("tiny")
        vocab = build_vocab_from_iterator(
            (tokenizer(x) for x in train_iter),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_freq=1,
            max_tokens=VOCAB_MAX_SIZE,
        )
    vocab.set_default_index(vocab["<unk>"])

    # Reload train/valid iterators (the first pass is exhausted), unless tiny
    if args.dataset == "ptb":
        train_iter = PennTreebank(split='train')
        valid_iter = PennTreebank(split='valid')
    elif args.dataset == "wikitext2":
        train_iter = WikiText2(split='train')
        valid_iter = WikiText2(split='valid')
    # Build token id tensors
    if args.dataset == "tiny":
        train_ids = encode_corpus(train_iter, vocab, tokenizer)
        valid_ids = encode_corpus(valid_iter, vocab, tokenizer)
    else:
        train_ids = encode_corpus(train_iter, vocab, tokenizer)
        valid_ids = encode_corpus(valid_iter, vocab, tokenizer)

    print(f"Vocab size (capped): {len(vocab)} | Train tokens: {len(train_ids)} | Valid tokens: {len(valid_ids)}")

    train_ds = LMSequenceDataset(train_ids, BLOCK_SIZE)
    valid_ds = LMSequenceDataset(valid_ids, BLOCK_SIZE)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=batchify)
    valid_loader = DataLoader(valid_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=batchify)

    # Classical model
    print("\n=== Training Classical Transformer LM ===")
    classical = ClassicalTransformerLM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        max_seq_len=MAX_SEQ_LEN,
        num_heads=NUM_HEADS,
    )
    classical_res = train_one_model(classical, train_loader, valid_loader, EPOCHS)

    # Quantum-embedding model
    print("\n=== Training Quantum-Embedding Transformer LM ===")
    quantum = QuantumEmbeddingTransformerLM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        max_seq_len=MAX_SEQ_LEN,
        num_heads=NUM_HEADS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        shots=SHOTS,
    )
    quantum_res = train_one_model(quantum, train_loader, valid_loader, EPOCHS)

    # Summaries
    ch = classical_res["history"]
    qh = quantum_res["history"]
    print("\n=== Summary ===")
    print(
        f"Classical - Final Val Loss: {ch['val_loss'][-1]:.4f}, PPL: {ch['val_ppl'][-1]:.2f}\n"
        f"Quantum   - Final Val Loss: {qh['val_loss'][-1]:.4f}, PPL: {qh['val_ppl'][-1]:.2f}"
    )

    # Generation
    prompt = "the meaning of life"
    print("\n--- Generation (Classical) ---")
    print(generate(classical_res["model"], vocab, prompt, max_new_tokens=30))

    print("\n--- Generation (Quantum Embedding) ---")
    print(generate(quantum_res["model"], vocab, prompt, max_new_tokens=30))


if __name__ == "__main__":
    main()
