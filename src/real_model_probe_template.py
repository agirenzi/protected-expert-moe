"""
Real-model Phase-1 probe template (TransformerLens / Hugging Face).

This is a clean, complete template rather than a fully automatic script
because:

- It requires a GPU and the actual model weights.
- Prompt sets and exact layer / hook choices must be chosen by the experimenter.
- Statistical thresholds need calibration on your data.

Typical usage on a machine with GPU + ~20 GB VRAM:

    pip install transformer-lens transformers accelerate
    huggingface-cli login          # if the model is gated
    python real_model_probe_template.py
"""

from __future__ import annotations

from typing import List, Tuple

import torch

# Optional imports – fail gracefully when missing
try:
    from transformer_lens import HookedTransformer
    HAS_TL = True
except ImportError:
    HAS_TL = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False


def load_model_tl(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load with TransformerLens (preferred for clean activation hooks)."""
    if not HAS_TL:
        raise RuntimeError("transformer_lens is required for this path")
    model = HookedTransformer.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    return model


def collect_swiglu_activations(
    model,
    prompts: List[str],
    layer: int,
    hook_name: str = "hook_mid",
) -> torch.Tensor:
    """
    Collect intermediate activations from a SwiGLU / MLP block.

    Exact hook name depends on the model; inspect model.hook_dict.keys()
    or the corresponding Hugging Face module names.
    Returns a tensor that can be ranked by the differential probe.
    """
    acts = []

    def hook_fn(activation, hook):
        acts.append(activation.detach().cpu())
        return activation

    for prompt in prompts:
        model.reset_hooks()
        _ = model.run_with_hooks(
            prompt,
            fwd_hooks=[(f"blocks.{layer}.mlp.{hook_name}", hook_fn)],
        )
    # Note: for production use you will want proper padding / masking
    # when sequences have different lengths.
    return torch.cat(acts, dim=0)


def rank_differential_channels(
    acts_harmful: torch.Tensor,
    acts_benign: torch.Tensor,
    top_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple ranking used in many activation-outlier studies:
    combination of max-abs ratio and a kurtosis-style term.
    """
    max_h = acts_harmful.abs().amax(dim=(0, 1))
    max_b = acts_benign.abs().amax(dim=(0, 1))
    ratio = max_h / (max_b + 1e-6)

    # Rough kurtosis proxy (higher = heavier tails)
    centered = acts_harmful - acts_harmful.mean(dim=(0, 1), keepdim=True)
    kurt = (centered ** 4).mean(dim=(0, 1)) / (acts_harmful.var(dim=(0, 1)) ** 2 + 1e-6)

    score = ratio * torch.log1p(kurt)
    vals, idx = torch.topk(score, k=top_k)
    return idx, vals


def ablation_test(
    model,
    prompts: List[str],
    layer: int,
    channels: List[int],
    hook_name: str = "hook_mid",
) -> List[str]:
    """
    Zero-ablate the listed channels and collect generations.
    Compare refusal rate against the un-ablated baseline.
    """
    def ablate_hook(activation, hook):
        activation[..., channels] = 0.0
        return activation

    outputs = []
    for prompt in prompts:
        model.reset_hooks()
        # For real generation you would normally call model.generate
        # with the hook registered.  The exact call depends on whether
        # you stay inside TransformerLens or drop back to HF generate.
        outputs.append("<ablated generation placeholder>")
    return outputs


if __name__ == "__main__":
    if not (HAS_TL or HAS_HF):
        print("Neither transformer_lens nor transformers is available.")
        print("This file is a template for use on a properly equipped machine.")
        raise SystemExit(0)

    print("Template loaded successfully.")
    print("Fill in real prompt lists, choose layers, and run on GPU.")
    print()
    print("Suggested workflow:")
    print("  1. model = load_model_tl()")
    print("  2. Collect acts_harmful and acts_benign with collect_swiglu_activations")
    print("  3. top_channels, scores = rank_differential_channels(...)")
    print("  4. Run ablation_test and measure refusal / capability deltas")
    print("  5. Decide whether the signal is strong enough to protect those channels")
