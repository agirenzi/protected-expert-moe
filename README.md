# Protected Expert MoE

# Protected Expert MoE

A minimal, fully working research prototype for isolating high-precision experts from aggressive post-training quantization, together with a clean experimental protocol for studying whether certain SwiGLU activation outliers carry safety-relevant signal.

[![Share on X](https://img.shields.io/badge/Share%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white)](https://twitter.com/intent/tweet?text=Protected%20Expert%20MoE%20%E2%80%94%20a%20research%20prototype%20for%20isolating%20high-precision%20experts%20%26%20studying%20SwiGLU%20activation%20outliers%20with%20safety%20signal&url=https%3A%2F%2Fgithub.com%2Fagirenzi%2Fprotected-expert-moe)

This package is intentionally small, self-contained, and free of incomplete fragments. Everything that claims to run has been executed successfully.

## Why this exists

Modern LLMs that use SwiGLU produce heavy-tailed activation distributions. The largest spikes interact with normalization layers and become problematic under low-precision training and post-training quantization. At the same time, mechanistic interpretability work shows that a sparse set of high-magnitude channels can be causally important for particular behaviors, including refusal.

The practical question is therefore:

> Can we measure which channels show differential spikes on safety-relevant inputs, and then protect a small high-precision path so that refusal behavior survives aggressive quantization of the rest of the model?

This repository supplies:

1. A clean experimental protocol for answering that question.
2. A correct, vectorized PyTorch implementation of a protected-expert MoE layer with hard override and explicit dtype enforcement.
3. A synthetic probe + harness that recovers injected ground-truth channels and measures override behavior.
4. A ready-to-adapt template for real-model probing with TransformerLens.
5. Example quantization exclusion configurations.

## Quick start (CPU, no model weights required)

```bash
cd src
python protected_expert_moe.py
python synthetic_probe_and_harness.py
```

Both scripts complete in well under a second and print clear diagnostics. The synthetic probe recovers the exact channels that were injected; the routing harness shows the hard override firing preferentially on the safety-like patterns.

## Repository layout

```
protected-expert-moe/
├── README.md                          # this file
├── requirements.txt
├── src/
│   ├── protected_expert_moe.py        # core MoE layer (runnable)
│   ├── synthetic_probe_and_harness.py # synthetic differential probe + eval
│   └── real_model_probe_template.py   # TransformerLens / HF template
├── configs/
│   └── quantization_exclusion_examples.json
└── docs/
    └── experimental_protocol.md       # full Phase-1 / Phase-2 protocol
```

## Design decisions (important)

- Fully vectorized dispatch — no per-token Python loops.
- Explicit `assert_protected_precision()` so a quantization tool cannot silently change the protected path.
- Hard override is a simple boolean mask derived from routing probability.
- The synthetic experiment has a clear ground-truth success condition.
- No claim is made that activation spikes “are” the ethical framework. The code only implements measurement and isolation; any causal interpretation must come from the ablation and quantization experiments in the protocol.

## Experimental protocol (summary)

**Phase 1 – Diagnostic**  
Hook intermediate SwiGLU activations on an aligned model, rank channels by differential spike statistics on matched harmful vs benign prompts, ablate the top candidates, and compare against random and high-magnitude controls. Then measure how much refusal degrades under unprotected vs channel-protected quantization.

**Phase 2 – Isolation**  
Replace a small number of FFN blocks with `ProtectedExpertMoE`, keep the router + protected expert in high precision, quantize everything else, and measure the recovery of refusal rate relative to the unprotected baseline.

Full details, success criteria, and statistical recommendations are in `docs/experimental_protocol.md`.

## Requirements

Minimal (synthetic experiments):

```
torch >= 2.0
```

Real-model probing (optional):

```
transformer-lens
transformers
accelerate
```

See `requirements.txt`.

## Limitations

- The MoE implementation is a research prototype, not a production serving kernel.
- Real causal claims require the ablation and quantization experiments; the synthetic probe only demonstrates the measurement technique.
- Router attack surface, capability leakage into protected channels, and distribution shift between probe sets and real jailbreaks remain open experimental questions.

## License

MIT (see below). Use, modify, and share freely. Attribution appreciated but not required.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
