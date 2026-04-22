# QuantumAttention

This repository explores a **hybrid quantum\-classical transformer** where classical embedding and attention layers can be replaced with quantum counterparts. The repo now supports multiple interchangeable language-model variants rather than a single fixed pipeline.

```
Input Tokens → {Classical or Quantum Embedding} → {Classical or Quantum Attention} → Classical Output Head
```

Key components include a quantum embedding layer implemented with **parameterized quantum circuits (PQCs)**, a hybrid attention mechanism and standard transformer blocks. The goal is to investigate how quantum features might enrich language models on near term (NISQ) hardware.

## Quantum Embedding Layer

Key design details:

- Qubits required: `n_q = ceil(log2(d_model))`
- Angle encoding and 3\-5 layers of RY/RZ rotations with CNOT entanglement
- Pauli\-Z measurements provide features which are transformed via a small linear layer
- Gradients are computed using the parameter\-shift rule with a shot budget of roughly 1000\-5000

The implementation can be found in [`src/quantum_embedding.py`](src/quantum_embedding.py). The file defines a basic `QuantumEmbedding` and an `OptimizedQuantumEmbedding` that caches quantum evaluations for efficiency.

## Quantum Attention Layer

The quantum attention implementation lives in [`src/quantum_attention.py`](src/quantum_attention.py).

- `QuantumAttention`: single-head attention with two score backends:
  - `swap_test`: PennyLane-backed overlap estimation
  - `fidelity`: fast PyTorch overlap approximation for development and benchmarking
- `HybridMultiHeadAttention`: true multi-head wrapper that splits the embedding dimension across heads, runs one quantum head per split, then merges results with an output projection.
- Supports transformer-style `attn_mask` and `key_padding_mask`, including causal masking for autoregressive language models.
- Returned attention weights are averaged across heads by default for a stable `(seq_len, batch_size, seq_len)` interface.

## Language Model Variants

The language-model stack in [`src/transformer_lm.py`](src/transformer_lm.py) now supports four benchmarkable variants:

- `ClassicalTransformerLM`
- `QuantumEmbeddingTransformerLM`
- `QuantumAttentionTransformerLM`
- `QuantumEmbeddingQuantumAttentionTransformerLM`

This makes it possible to compare embedding-only, attention-only, and fully hybrid quantum variants side by side.

## Training Comparison

*   `train_compare.py`:
    *   Compares `ClassicalTransformer` vs `HybridQuantumEmbeddingTransformer` on a synthetic sentiment classification task.
    *   Generates `quantum_embedding_comparison.png` with train/validation loss and accuracy curves.
    *   Runtime is typically much higher for the quantum-enhanced model due to circuit simulation.
    *   Winner depends on the exact run configuration (data seed, shots, epochs, and model size).

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Then run the example training:

```bash
python train_quantum_embedding.py
```

This will train both models for a few epochs and save the comparison plot.

## Reproducible Evaluation

Run the full test suite:

```bash
pytest -q
```

Run language-model comparison on a real corpus:

```bash
python train_lm_generation_compare.py --dataset ptb --epochs 2
```

Run the synthetic classification benchmark with quantum attention:

```bash
python train_quantum_attention_compare.py --epochs 10 --models classical,quantum_embedding,quantum_attention,full_quantum --score_mode fidelity
```

Run the unified cross-task benchmark report:

```bash
python3 train_unified_benchmark_report.py --models classical,quantum_embedding,quantum_attention,full_quantum --qa_models quantum_attention,full_quantum --primary_score_mode fidelity --compare_score_modes fidelity,swap_test --seed 42 --output_dir artifacts/unified_benchmark --summary_plot_path unified_benchmark_summary.png --summary_metrics_path artifacts/unified_benchmark_summary.json --summary_markdown_path artifacts/unified_benchmark_summary.md --lm_dataset tiny --lm_epochs 2 --lm_train_steps 8 --lm_eval_steps 4 --lm_vocab_max 200 --lm_block 4 --classification_epochs 2 --classification_batch_size 8 --classification_train_size 32 --classification_val_size 16 --classification_seq_len 6 --classification_vocab_size 20 --classification_embed_dim 8 --classification_n_qubits 3 --classification_n_layers 1 --classification_shots 20
```

Useful options:

- `--dataset wikitext2` (download on first run)
- `--dataset tiny` (offline fallback, fastest)
- `--models classical,quantum_embedding,quantum_attention,full_quantum`
- `--score_mode fidelity` for the fast benchmark path, or `--score_mode swap_test` for the quantum SWAP-test path
- `--metrics_path artifacts/lm_benchmark_metrics.json` to save structured benchmark output

The LM benchmark harness records:

- train/validation loss and perplexity
- per-epoch timing
- parameter counts
- generations for each selected model
- plot output plus JSON metrics artifacts

The classification benchmark records the same style of plot and JSON metrics for the synthetic sentiment task.

The unified benchmark report writes:

- `artifacts/unified_benchmark/lm_primary_metrics.json`
- `artifacts/unified_benchmark/classification_primary_metrics.json`
- `artifacts/unified_benchmark_summary.json`
- `artifacts/unified_benchmark_summary.md`
- `unified_benchmark_summary.png`

The score-mode comparison in the unified report is controlled: it evaluates `fidelity` and `swap_test` on the **same trained quantum-attention weights** instead of retraining separate models with different backends.

CI is configured in `.github/workflows/ci.yml` and runs tests on Python 3.10 and 3.11 for pushes and pull requests.

## Unified Benchmark Snapshot

Last updated: **2026-04-22**

Benchmark command:

```bash
python3 train_unified_benchmark_report.py --models classical,quantum_embedding,quantum_attention,full_quantum --qa_models quantum_attention,full_quantum --primary_score_mode fidelity --compare_score_modes fidelity,swap_test --seed 42 --output_dir artifacts/unified_benchmark --summary_plot_path unified_benchmark_summary.png --summary_metrics_path artifacts/unified_benchmark_summary.json --summary_markdown_path artifacts/unified_benchmark_summary.md --lm_dataset tiny --lm_epochs 2 --lm_train_steps 8 --lm_eval_steps 4 --lm_vocab_max 200 --lm_block 4 --classification_epochs 2 --classification_batch_size 8 --classification_train_size 32 --classification_val_size 16 --classification_seq_len 6 --classification_vocab_size 20 --classification_embed_dim 8 --classification_n_qubits 3 --classification_n_layers 1 --classification_shots 20
```

Primary results from the latest run (CPU):

| Variant | LM Final Val PPL | Classification Final Val Acc |
| --- | ---: | ---: |
| Classical | 9.30 | 0.3750 |
| Quantum Embedding | 9.59 | 0.3750 |
| Quantum Attention | 9.10 | 0.3750 |
| Full Hybrid | 5.16 | 0.6250 |

Controlled score-mode evaluation on the same trained quantum-attention weights:

| Variant | LM Fidelity PPL | LM Swap-Test PPL | LM Swap/Fidelity Eval Time | Classification Fidelity Acc | Classification Swap-Test Acc | Classification Swap/Fidelity Eval Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantum Attention | 9.10 | 9.09 | 580.32x | 0.3750 | 0.3750 | 1182.89x |
| Full Hybrid | 5.16 | 5.16 | 112.76x | 0.6250 | 0.6250 | 82.49x |

Latest unified benchmark plot (same run as tables above):

![Latest unified benchmark plot](unified_benchmark_summary.png)

Interpretation from this snapshot:

- The full hybrid model is the strongest overall variant on both tasks.
- Attention-only quantum attention improves the tiny LM validation perplexity over the classical baseline, but it does not improve the synthetic classification accuracy in this run.
- `swap_test` closely matches `fidelity` on validation metrics here, but it is dramatically slower at evaluation time.

## Latest Classification Snapshot

Last updated: **2026-04-03**

Benchmark command:

```bash
python3 train_quantum_attention_compare.py --epochs 3 --batch_size 8 --train_size 32 --val_size 16 --seq_len 6 --vocab_size 20 --embed_dim 8 --n_qubits 3 --n_layers 1 --shots 20 --models classical,quantum_embedding,quantum_attention,full_quantum --score_mode fidelity --plot_path quantum_attention_classification_comparison.png --metrics_path artifacts/classification_benchmark_metrics.json
```

Results from the latest run (CPU):

| Model | Final Train Loss | Final Val Loss | Final Val Acc |
| --- | ---: | ---: | ---: |
| ClassicalTransformer | 0.7103 | 0.7132 | 0.3750 |
| HybridQuantumEmbeddingTransformer | 0.6794 | 0.6731 | 0.6250 |
| QuantumAttentionTransformer | 0.6683 | 0.6599 | 0.6250 |
| QuantumEmbeddingQuantumAttentionTransformer | 0.6908 | 0.6881 | 0.6250 |

Latest classification benchmark plot (same run as table above):

![Latest classification benchmark plot](quantum_attention_classification_comparison.png)

## Latest Performance Snapshot

Last updated: **2026-04-02**

Benchmark command:

```bash
python train_lm_generation_compare.py --dataset tiny --epochs 3 --train_steps 80 --eval_steps 20 --vocab_max 200 --block 4 --seed 42 --models classical,quantum_embedding,quantum_attention,full_quantum --score_mode fidelity --plot_path lm_generation_comparison.png --metrics_path artifacts/lm_benchmark_metrics.json
```

Results from the latest run (CPU):

| Model | Final Train Loss | Final Train PPL | Final Val Loss | Final Val PPL |
| --- | ---: | ---: | ---: | ---: |
| ClassicalTransformerLM | 2.1960 | 8.99 | 2.1701 | 8.76 |
| QuantumEmbeddingTransformerLM | 2.0771 | 7.98 | 2.0625 | 7.87 |
| QuantumAttentionTransformerLM | 2.4421 | 11.50 | 2.4899 | 12.06 |
| QuantumEmbeddingQuantumAttentionTransformerLM | 2.1526 | 8.61 | 2.2211 | 9.22 |

Latest benchmark plot (same run as table above):

![Latest LM benchmark plot](lm_generation_comparison.png)

Note: `tiny` is an offline sanity benchmark intended for fast iteration, not a final quality benchmark.

Consistency rule: whenever metrics are updated, plots must be regenerated from the same run and the README date/command/results must be updated together in one commit.
