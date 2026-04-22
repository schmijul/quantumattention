# Unified Benchmark Summary

## Run Metadata
- **seed**: `42`
- **primary_score_mode**: `fidelity`
- **compare_score_modes**: `fidelity,swap_test`
- **models**: `classical,quantum_embedding,quantum_attention,full_quantum`
- **qa_models**: `quantum_attention,full_quantum`
- **output_dir**: `artifacts/unified_benchmark`

## Language Modeling
Primary metric: **Final Val PPL** (lower is better)

| Variant | Final Val PPL | Final Val Loss | Avg Epoch (s) | Params | Delta vs Classical |
| --- | --- | --- | --- | --- | --- |
| Classical | 9.30 | 2.2296 | 0.08 | 7,640 | 0.00 |
| Quantum Embedding | 9.59 | 2.2611 | 2.09 | 7,816 | 0.30 |
| Quantum Attention | 9.10 | 2.2078 | 0.07 | 7,640 | -0.20 |
| Full Hybrid | 5.16 | 1.6407 | 3.07 | 7,816 | -4.14 |

Winner: **Full Hybrid**

## Classification
Primary metric: **Final Val Acc** (higher is better)

| Variant | Final Val Acc | Final Val Loss | Avg Epoch (s) | Params | Delta vs Classical |
| --- | --- | --- | --- | --- | --- |
| Classical | 0.3750 | 0.7211 | 0.02 | 610 | 0.0000 |
| Quantum Embedding | 0.3750 | 0.7443 | 1.50 | 602 | 0.0000 |
| Quantum Attention | 0.3750 | 0.7086 | 0.04 | 610 | 0.0000 |
| Full Hybrid | 0.6250 | 0.6828 | 1.51 | 602 | 0.2500 |

Winner: **Full Hybrid**

## Language Modeling Score-Mode Comparison
| Variant | Fidelity | Swap Test | Fidelity Eval Time (s) | Swap Test Eval Time (s) | Swap Test / Fidelity Time |
| --- | --- | --- | --- | --- | --- |
| Quantum Attention | 9.10 | 9.09 | 0.00 | 1.60 | 580.32 |
| Full Hybrid | 5.16 | 5.16 | 0.01 | 1.50 | 112.76 |

## Classification Score-Mode Comparison
| Variant | Fidelity | Swap Test | Fidelity Eval Time (s) | Swap Test Eval Time (s) | Swap Test / Fidelity Time |
| --- | --- | --- | --- | --- | --- |
| Quantum Attention | 0.3750 | 0.3750 | 0.01 | 6.46 | 1182.89 |
| Full Hybrid | 0.6250 | 0.6250 | 0.07 | 5.98 | 82.49 |
