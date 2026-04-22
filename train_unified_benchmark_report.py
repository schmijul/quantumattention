"""Run both benchmark harnesses and aggregate them into one reproducible report."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import PennTreebank, WikiText2
from torchtext.vocab import build_vocab_from_iterator

import train_lm_generation_compare as lm_benchmark
import train_quantum_attention_compare as classification_benchmark
from src.benchmarking import (
    MODEL_LABEL_TO_VARIANT,
    TASK_SPECS,
    VARIANT_DISPLAY_NAMES,
    build_score_mode_comparison,
    build_unified_benchmark_summary,
    plot_unified_benchmark_summary,
    render_unified_benchmark_markdown,
    write_json,
    write_markdown,
)


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_metrics_payload(payload: Mapping[str, Any], variant_keys: Iterable[str]) -> Dict[str, Any]:
    variant_key_set = set(variant_keys)
    filtered_results = {
        label: result
        for label, result in payload["results"].items()
        if MODEL_LABEL_TO_VARIANT[label] in variant_key_set
    }
    config = dict(payload["config"])
    config["models"] = [model for model in config.get("models", []) if model in variant_key_set]
    return {
        "config": config,
        "results": filtered_results,
    }


def configure_lm_globals(args: argparse.Namespace) -> None:
    lm_benchmark.EPOCHS = args.lm_epochs
    lm_benchmark.TRAIN_STEPS_LIMIT = args.lm_train_steps
    lm_benchmark.EVAL_STEPS_LIMIT = args.lm_eval_steps
    lm_benchmark.VOCAB_MAX_SIZE = args.lm_vocab_max
    lm_benchmark.BLOCK_SIZE = args.lm_block


def prepare_lm_data(args: argparse.Namespace):
    configure_lm_globals(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_iter, valid_iter = lm_benchmark.load_iters(args.lm_dataset)
    tokenizer = get_tokenizer("basic_english")
    try:
        vocab = build_vocab_from_iterator(
            lm_benchmark.yield_tokens(train_iter, tokenizer)
            if args.lm_dataset != "tiny"
            else (tokenizer(text) for text in train_iter),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_freq=lm_benchmark.MIN_FREQ,
            max_tokens=lm_benchmark.VOCAB_MAX_SIZE,
        )
        dataset_name = args.lm_dataset
    except Exception as exc:
        print(f"Falling back to tiny offline dataset due to error: {exc}")
        dataset_name = "tiny"
        train_iter, valid_iter = lm_benchmark.load_iters("tiny")
        vocab = build_vocab_from_iterator(
            (tokenizer(text) for text in train_iter),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
            min_freq=1,
            max_tokens=lm_benchmark.VOCAB_MAX_SIZE,
        )

    vocab.set_default_index(vocab["<unk>"])

    if dataset_name == "ptb":
        train_iter = PennTreebank(split="train")
        valid_iter = PennTreebank(split="valid")
    elif dataset_name == "wikitext2":
        train_iter = WikiText2(split="train")
        valid_iter = WikiText2(split="valid")
    elif dataset_name == "tiny":
        train_iter, valid_iter = lm_benchmark.load_iters("tiny")

    train_ids = lm_benchmark.encode_corpus(train_iter, vocab, tokenizer)
    valid_ids = lm_benchmark.encode_corpus(valid_iter, vocab, tokenizer)

    train_ds = lm_benchmark.LMSequenceDataset(train_ids, lm_benchmark.BLOCK_SIZE)
    valid_ds = lm_benchmark.LMSequenceDataset(valid_ids, lm_benchmark.BLOCK_SIZE)
    train_loader = DataLoader(
        train_ds,
        batch_size=lm_benchmark.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lm_benchmark.batchify,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=lm_benchmark.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=lm_benchmark.batchify,
    )
    return dataset_name, vocab, train_loader, valid_loader


def run_lm_benchmark(
    args: argparse.Namespace,
    models: List[str],
    score_mode: str,
    plot_path: Path,
    metrics_path: Path,
) -> Dict[str, Any]:
    dataset_name, vocab, train_loader, valid_loader = prepare_lm_data(args)
    factories = lm_benchmark.build_model_factories(len(vocab), score_mode)
    results: Dict[str, Any] = {}
    generations: Dict[str, str] = {}
    prompt = "the meaning of life"

    print(f"Using device: {lm_benchmark.DEVICE}")
    print(f"Dataset: {dataset_name}")
    print(
        f"Vocab size: {len(vocab)} | Train batches: {len(train_loader)} | "
        f"Valid batches: {len(valid_loader)}"
    )

    for key in models:
        label, factory = factories[key]
        print(f"\n=== Training {label} ===")
        run_result = lm_benchmark.train_one_model(label, factory(), train_loader, valid_loader, args.lm_epochs)
        results[label] = run_result
        generations[label] = lm_benchmark.generate(
            run_result["model"],
            vocab,
            prompt,
            lm_benchmark.BLOCK_SIZE,
            max_new_tokens=30,
        )

    lm_benchmark.plot_histories(results, str(plot_path))
    payload = {
        "config": {
            "dataset": dataset_name,
            "epochs": args.lm_epochs,
            "train_steps": args.lm_train_steps,
            "eval_steps": args.lm_eval_steps,
            "vocab_max": args.lm_vocab_max,
            "block_size": args.lm_block,
            "embed_dim": lm_benchmark.EMBED_DIM,
            "num_heads": lm_benchmark.NUM_HEADS,
            "max_seq_len": lm_benchmark.MAX_SEQ_LEN,
            "n_qubits": lm_benchmark.N_QUBITS,
            "n_layers": lm_benchmark.N_LAYERS,
            "shots": lm_benchmark.SHOTS,
            "seed": args.seed,
            "device": str(lm_benchmark.DEVICE),
            "score_mode": score_mode,
            "models": models,
            "plot_path": str(plot_path),
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
    write_json(metrics_path, payload)
    return {
        "payload": payload,
        "results": results,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "vocab_size": len(vocab),
    }


def configure_classification_globals(args: argparse.Namespace) -> None:
    classification_benchmark.BATCH_SIZE = args.classification_batch_size
    classification_benchmark.EMBED_DIM = args.classification_embed_dim
    classification_benchmark.SEQ_LEN = args.classification_seq_len
    classification_benchmark.VOCAB_SIZE = args.classification_vocab_size
    classification_benchmark.TRAIN_SIZE = args.classification_train_size
    classification_benchmark.VAL_SIZE = args.classification_val_size
    classification_benchmark.N_QUBITS = args.classification_n_qubits
    classification_benchmark.N_LAYERS = args.classification_n_layers
    classification_benchmark.SHOTS = args.classification_shots
    classification_benchmark.N_EPOCHS = args.classification_epochs


def prepare_classification_data(args: argparse.Namespace):
    configure_classification_globals(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    return classification_benchmark.make_loaders(args.seed)


def run_classification_benchmark(
    args: argparse.Namespace,
    models: List[str],
    score_mode: str,
    plot_path: Path,
    metrics_path: Path,
) -> Dict[str, Any]:
    train_loader, val_loader = prepare_classification_data(args)
    factories = classification_benchmark.build_factories(score_mode)
    results: Dict[str, Any] = {}

    print(f"Using device: {classification_benchmark.DEVICE}")
    for key in models:
        label, factory = factories[key]
        results[label] = classification_benchmark.run_experiment(
            label,
            factory,
            train_loader,
            val_loader,
            args.classification_epochs,
        )

    classification_benchmark.plot_results(results, str(plot_path))
    payload = {
        "config": {
            "epochs": args.classification_epochs,
            "batch_size": args.classification_batch_size,
            "embed_dim": args.classification_embed_dim,
            "seq_len": args.classification_seq_len,
            "vocab_size": args.classification_vocab_size,
            "train_size": args.classification_train_size,
            "val_size": args.classification_val_size,
            "n_qubits": args.classification_n_qubits,
            "n_layers": args.classification_n_layers,
            "shots": args.classification_shots,
            "seed": args.seed,
            "device": str(classification_benchmark.DEVICE),
            "score_mode": score_mode,
            "models": models,
            "plot_path": str(plot_path),
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
    write_json(metrics_path, payload)
    return {
        "payload": payload,
        "results": results,
        "train_loader": train_loader,
        "valid_loader": val_loader,
    }


def build_lm_score_mode_summaries(
    args: argparse.Namespace,
    trained_results: Mapping[str, Any],
    valid_loader: DataLoader,
    vocab_size: int,
    qa_models: List[str],
    compare_score_modes: List[str],
) -> Dict[str, Dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    summaries: Dict[str, Dict[str, Any]] = {}
    for score_mode in compare_score_modes:
        factories = lm_benchmark.build_model_factories(vocab_size, score_mode)
        rows = []
        for variant_key in qa_models:
            model_label, factory = factories[variant_key]
            model = factory().to(lm_benchmark.DEVICE)
            model.load_state_dict(trained_results[model_label]["model"].state_dict())
            start = time.perf_counter()
            val_loss, val_ppl = lm_benchmark.evaluate(
                model,
                valid_loader,
                criterion,
                steps_limit=args.lm_eval_steps,
            )
            eval_time = time.perf_counter() - start
            rows.append(
                {
                    "variant_key": variant_key,
                    "model_label": model_label,
                    "display_name": VARIANT_DISPLAY_NAMES[variant_key],
                    "final_val_loss": val_loss,
                    "final_val_ppl": val_ppl,
                    "eval_time_sec": eval_time,
                }
            )
        summaries[score_mode] = {
            "task": "language_modeling",
            **TASK_SPECS["language_modeling"],
            "rows": rows,
        }
    return summaries


def build_classification_score_mode_summaries(
    trained_results: Mapping[str, Any],
    valid_loader: DataLoader,
    qa_models: List[str],
    compare_score_modes: List[str],
) -> Dict[str, Dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    summaries: Dict[str, Dict[str, Any]] = {}
    for score_mode in compare_score_modes:
        factories = classification_benchmark.build_factories(score_mode)
        rows = []
        for variant_key in qa_models:
            model_label, factory = factories[variant_key]
            model = factory().to(classification_benchmark.DEVICE)
            model.load_state_dict(trained_results[model_label]["model"].state_dict())
            start = time.perf_counter()
            val_loss, val_acc = classification_benchmark.eval_epoch(model, valid_loader, criterion)
            eval_time = time.perf_counter() - start
            rows.append(
                {
                    "variant_key": variant_key,
                    "model_label": model_label,
                    "display_name": VARIANT_DISPLAY_NAMES[variant_key],
                    "final_val_loss": val_loss,
                    "final_val_acc": val_acc,
                    "eval_time_sec": eval_time,
                }
            )
        summaries[score_mode] = {
            "task": "classification",
            **TASK_SPECS["classification"],
            "rows": rows,
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both benchmark harnesses and write one unified report")
    parser.add_argument(
        "--models",
        type=str,
        default="classical,quantum_embedding,quantum_attention,full_quantum",
        help="Comma-separated model keys for the primary four-way comparison",
    )
    parser.add_argument(
        "--qa_models",
        type=str,
        default="quantum_attention,full_quantum",
        help="Comma-separated model keys for the controlled score-mode comparison",
    )
    parser.add_argument(
        "--primary_score_mode",
        choices=["swap_test", "fidelity"],
        default="fidelity",
        help="Score backend for the primary four-way comparison",
    )
    parser.add_argument(
        "--compare_score_modes",
        type=str,
        default="fidelity,swap_test",
        help="Comma-separated score backends to evaluate on the same trained quantum-attention weights",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="artifacts/unified_benchmark")
    parser.add_argument("--summary_plot_path", type=str, default="unified_benchmark_summary.png")
    parser.add_argument("--summary_metrics_path", type=str, default="artifacts/unified_benchmark_summary.json")
    parser.add_argument("--summary_markdown_path", type=str, default="artifacts/unified_benchmark_summary.md")

    parser.add_argument("--lm_dataset", choices=["ptb", "wikitext2", "tiny"], default="tiny")
    parser.add_argument("--lm_epochs", type=int, default=3)
    parser.add_argument("--lm_train_steps", type=int, default=80)
    parser.add_argument("--lm_eval_steps", type=int, default=20)
    parser.add_argument("--lm_vocab_max", type=int, default=200)
    parser.add_argument("--lm_block", type=int, default=4)

    parser.add_argument("--classification_epochs", type=int, default=3)
    parser.add_argument("--classification_batch_size", type=int, default=8)
    parser.add_argument("--classification_train_size", type=int, default=32)
    parser.add_argument("--classification_val_size", type=int, default=16)
    parser.add_argument("--classification_seq_len", type=int, default=6)
    parser.add_argument("--classification_vocab_size", type=int, default=20)
    parser.add_argument("--classification_embed_dim", type=int, default=8)
    parser.add_argument("--classification_n_qubits", type=int, default=3)
    parser.add_argument("--classification_n_layers", type=int, default=1)
    parser.add_argument("--classification_shots", type=int, default=20)
    args = parser.parse_args()

    models = parse_csv(args.models)
    qa_models = parse_csv(args.qa_models)
    compare_score_modes = parse_csv(args.compare_score_modes)
    if not compare_score_modes:
        raise ValueError("compare_score_modes must include at least one score backend")
    if not set(qa_models).issubset(models):
        raise ValueError("qa_models must be a subset of the primary models")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_lm = run_lm_benchmark(
        args,
        models=models,
        score_mode=args.primary_score_mode,
        plot_path=output_dir / "lm_primary_plot.png",
        metrics_path=output_dir / "lm_primary_metrics.json",
    )
    primary_classification = run_classification_benchmark(
        args,
        models=models,
        score_mode=args.primary_score_mode,
        plot_path=output_dir / "classification_primary_plot.png",
        metrics_path=output_dir / "classification_primary_metrics.json",
    )

    lm_payload = primary_lm["payload"]
    classification_payload = primary_classification["payload"]
    summary = build_unified_benchmark_summary(
        lm_payload=lm_payload,
        classification_payload=classification_payload,
        score_mode_payloads={
            "language_modeling": {
                args.primary_score_mode: filter_metrics_payload(lm_payload, qa_models),
            },
            "classification": {
                args.primary_score_mode: filter_metrics_payload(classification_payload, qa_models),
            },
        },
        metadata={
            "seed": args.seed,
            "primary_score_mode": args.primary_score_mode,
            "compare_score_modes": ",".join(compare_score_modes),
            "models": ",".join(models),
            "qa_models": ",".join(qa_models),
            "output_dir": str(output_dir),
        },
    )

    score_mode_comparisons = {
        "language_modeling": build_score_mode_comparison(
            "language_modeling",
            build_lm_score_mode_summaries(
                args,
                primary_lm["results"],
                primary_lm["valid_loader"],
                primary_lm["vocab_size"],
                qa_models,
                compare_score_modes,
            ),
        ),
        "classification": build_score_mode_comparison(
            "classification",
            build_classification_score_mode_summaries(
                primary_classification["results"],
                primary_classification["valid_loader"],
                qa_models,
                compare_score_modes,
            ),
        ),
    }
    summary["score_mode_comparisons"] = score_mode_comparisons

    summary_metrics_path = write_json(args.summary_metrics_path, summary)
    summary_markdown_path = write_markdown(
        args.summary_markdown_path,
        render_unified_benchmark_markdown(summary),
    )
    summary_plot_path = plot_unified_benchmark_summary(summary, args.summary_plot_path)

    print(f"\nSaved unified metrics: {summary_metrics_path}")
    print(f"Saved unified markdown: {summary_markdown_path}")
    print(f"Saved unified plot: {summary_plot_path}")


if __name__ == "__main__":
    main()
