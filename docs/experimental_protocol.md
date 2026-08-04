# Experimental Protocol: Mapping & Protecting Safety-Relevant SwiGLU Outliers

## Core Hypotheses (Falsifiable)

1. A sparse subset of SwiGLU intermediate channels shows reliably elevated activation magnitude on safety-refusal prompts relative to closely matched benign prompts.
2. Selectively ablating or aggressively quantizing those channels reduces refusal rate more than it reduces general capability.
3. Isolating residual flow through a high-precision expert (triggered by those channels or by an explicit router) can preserve refusal under heavy quantization of the remainder of the model.

## Materials

- **Primary model**: `meta-llama/Meta-Llama-3-8B-Instruct` (full-precision BF16 baseline). Optional secondary checks: Mistral-7B-Instruct or Qwen2.5-7B-Instruct.
- **Hardware**: 1–2× A100/H100 80 GB (or equivalent).
- **Libraries**: `transformer-lens`, `transformers`, `bitsandbytes` / `auto-gptq` / `llm-awq`, `datasets`, `torch`.
- **Prompt resources**:
  - Safety set: 300–500 items from AdvBench + HarmBench categories, each paired with a closely matched benign rewrite.
  - Capability control: 200–300 items from MMLU (stratified) + short GSM8K-style items.
  - Robustness set: 100 multi-turn or obfuscated jailbreak attempts.

## Phase 1 — Diagnostic Mapping

### 1.1 Baseline collection
Load the model in BF16 with TransformerLens. Register forward hooks on intermediate SwiGLU activations for middle-to-late layers (e.g. 12–28). Run the full safety + matched-benign pairs. Store per-token, per-channel statistics and residual-stream snapshots.

### 1.2 Outlier identification
For each channel *c* in each layer *L*:
- Compute spike ratio = max(|act| on harmful) / (max(|act| on matched benign) + ε)
- Compute a kurtosis-style statistic of the activation distribution.
- Rank channels by a combined score.

Retain the top 0.1–0.5 % of channels per layer (and their union) as the candidate set. Also retain a matched number of high-magnitude but non-differential channels as controls.

### 1.3 Causal ablation
- Zero-ablate the candidate channels and re-evaluate refusal + capability.
- Repeat with random channels of equal count and with the high-magnitude control channels.
- Use McNemar or paired bootstrap tests on refusal success/failure.

### 1.4 Quantization stress test
Create three quantized versions of the same base model:
- Unprotected (standard AWQ / GPTQ INT4 or FP8).
- Protected (same quantization but force the candidate channels / corresponding weights to remain in BF16).
- Random-protected control (protect an equal number of non-candidate high-magnitude channels).

Evaluate refusal, capability, and robustness on all three.

**Checkpoint**: Proceed to Phase 2 only if the candidate channels produce both elevated differential activation *and* a statistically reliable larger drop in refusal under ablation or unprotected quantization than the controls.

## Phase 2 — Isolation Architecture

### 2.1 Minimal protected expert
Replace the FFN of a small number of middle-to-late layers with a `ProtectedExpertMoE` (or equivalent):
- Router and protected expert stay in BF16.
- General experts may be quantized.
- Hard override when protected routing probability ≥ τ (sweep τ).

### 2.2 Weight seeding
Initialize the protected expert from the corresponding layers of the original aligned model (or from a small refusal-focused preference fine-tune).

### 2.3 Quantization of the hybrid model
Quantize only the general experts and non-protected layers. Measure memory, throughput, refusal rate, and capability against the three baselines defined above.

## Analysis Standards

- Report effect sizes and confidence intervals.
- Always include the random-channel and high-magnitude-control conditions.
- Log exact channel indices, layers, and thresholds for reproducibility.
- Document failure modes: capability leakage into the protected channels, router attack surface, distribution shift between probe and robustness sets.

## Success Criteria (Suggested)

- Phase 1: Candidate channels produce a substantially larger refusal drop under ablation/unprotected quantization than controls, while capability drop remains comparable or smaller.
- Phase 2: The hybrid protected model recovers a large fraction of the refusal-rate gap between full BF16 and the unprotected low-bit model, at modest memory overhead.
