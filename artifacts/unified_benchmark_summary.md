# Unified Benchmark Summary

## Run Metadata
- **seed**: `42`
- **primary_score_mode**: `fidelity`
- **compare_score_modes**: `fidelity,swap_test`
- **models**: `classical,quantum_embedding,quantum_attention,full_quantum`
- **qa_models**: `quantum_attention,full_quantum`
- **output_dir**: `artifacts/unified_benchmark`

## Cross-Task Overview
Combined ranking uses the mean percent improvement vs the classical baseline across both tasks.

| Variant | LM Final Val PPL | LM vs Classical | Classification Final Val Acc | Classification vs Classical | Task Wins | Mean Improvement vs Classical |
| --- | --- | --- | --- | --- | --- | --- |
| Full Hybrid | 5.16 | 44.51% | 0.6250 | 66.67% | 2 | 55.59% |
| Quantum Attention | 9.10 | 2.15% | 0.3750 | 0.00% | 0 | 1.08% |
| Classical | 9.30 | 0.00% | 0.3750 | 0.00% | 0 | 0.00% |
| Quantum Embedding | 9.59 | -3.20% | 0.3750 | 0.00% | 0 | -1.60% |

Overall leader: **Full Hybrid**

## Language Modeling
Primary metric: **Final Val PPL** (lower is better)

| Variant | Final Val PPL | Final Val Loss | Avg Epoch (s) | Params | Delta vs Classical |
| --- | --- | --- | --- | --- | --- |
| Classical | 9.30 | 2.2296 | 0.05 | 7,640 | 0.00 |
| Quantum Embedding | 9.59 | 2.2611 | 2.20 | 7,816 | 0.30 |
| Quantum Attention | 9.10 | 2.2078 | 0.07 | 7,640 | -0.20 |
| Full Hybrid | 5.16 | 1.6407 | 2.58 | 7,816 | -4.14 |

Winner: **Full Hybrid**

## Classification
Primary metric: **Final Val Acc** (higher is better)

| Variant | Final Val Acc | Final Val Loss | Avg Epoch (s) | Params | Delta vs Classical |
| --- | --- | --- | --- | --- | --- |
| Classical | 0.3750 | 0.7211 | 0.18 | 610 | 0.0000 |
| Quantum Embedding | 0.3750 | 0.7443 | 3.07 | 602 | 0.0000 |
| Quantum Attention | 0.3750 | 0.7086 | 0.06 | 610 | 0.0000 |
| Full Hybrid | 0.6250 | 0.6828 | 1.60 | 602 | 0.2500 |

Winner: **Full Hybrid**

## Language Modeling Score-Mode Comparison
| Variant | Fidelity | Swap Test | Fidelity Eval Time (s) | Swap Test Eval Time (s) | Swap Test / Fidelity Time |
| --- | --- | --- | --- | --- | --- |
| Quantum Attention | 9.10 | 9.09 | 0.00 | 2.75 | 995.23 |
| Full Hybrid | 5.16 | 5.16 | 0.03 | 2.94 | 94.92 |

## Classification Score-Mode Comparison
| Variant | Fidelity | Swap Test | Fidelity Eval Time (s) | Swap Test Eval Time (s) | Swap Test / Fidelity Time |
| --- | --- | --- | --- | --- | --- |
| Quantum Attention | 0.3750 | 0.3750 | 0.01 | 11.12 | 1581.59 |
| Full Hybrid | 0.6250 | 0.6250 | 0.12 | 10.35 | 86.35 |
