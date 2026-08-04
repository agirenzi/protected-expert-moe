"""
Synthetic differential-spike probe and routing robustness harness.

Demonstrates the diagnostic idea without requiring a real LLM:

1. Generate two families of residual-stream patterns
   ("benign-like" vs "safety-like").
2. Rank channels by a simple max-abs ratio statistic and verify that
   the injected differential channels are recovered.
3. Feed the same patterns through ProtectedExpertMoE and measure how
   often the hard override fires on each family.
"""

from __future__ import annotations

import torch
from protected_expert_moe import ProtectedExpertMoE


def make_synthetic_activations(
    batch: int,
    seq: int,
    d_model: int,
    n_spike_channels: int = 4,
    spike_strength: float = 8.0,
    safety: bool = False,
) -> torch.Tensor:
    """Create a residual-stream-like tensor.

    When safety=True a small contiguous set of channels receives large
    positive spikes (the ground-truth signal the probe should recover).
    """
    x = torch.randn(batch, seq, d_model) * 0.6
    if safety:
        spike = torch.zeros_like(x)
        spike[:, :, :n_spike_channels] = spike_strength
        spike += torch.randn_like(spike) * 0.3
        x = x + spike
    return x


def differential_spike_probe(
    benign: torch.Tensor,
    safety: torch.Tensor,
    top_k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank channels by safety / benign max-abs ratio."""
    max_b = benign.abs().amax(dim=(0, 1))
    max_s = safety.abs().amax(dim=(0, 1))
    ratio = max_s / (max_b + 1e-5)
    top_vals, top_idx = torch.topk(ratio, k=top_k)
    return top_idx, top_vals


def evaluate_routing(
    model: ProtectedExpertMoE,
    benign_x: torch.Tensor,
    safety_x: torch.Tensor,
) -> dict:
    model.eval()
    with torch.no_grad():
        _, info_b = model(benign_x, return_router_info=True)
        _, info_s = model(safety_x, return_router_info=True)

    return {
        "override_rate_benign": info_b["override_mask"].float().mean().item(),
        "override_rate_safety": info_s["override_mask"].float().mean().item(),
        "mean_protected_prob_benign": info_b["protected_prob"].mean().item(),
        "mean_protected_prob_safety": info_s["protected_prob"].mean().item(),
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"
    d_model = 128
    d_hidden = 256
    batch, seq = 8, 32

    benign = make_synthetic_activations(batch, seq, d_model, safety=False)
    safety = make_synthetic_activations(batch, seq, d_model, safety=True)

    # ----- Differential probe -----
    top_idx, top_vals = differential_spike_probe(benign, safety, top_k=6)
    print("=== Differential Spike Probe ===")
    print("Top channels by safety/benign max-abs ratio:")
    for i, (idx, val) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
        print(f"  rank {i}: channel {idx:3d}  ratio = {val:.2f}")

    recovered = set(top_idx[:4].tolist())
    expected = set(range(4))
    print(f"Recovered injected channels? {recovered == expected}  (recovered={sorted(recovered)})")

    # ----- Routing harness -----
    print("\n=== Routing Robustness Harness ===")
    for thr in (0.40, 0.55, 0.70):
        model = ProtectedExpertMoE(
            d_model=d_model,
            d_hidden=d_hidden,
            num_general_experts=3,
            protected_dtype=torch.float32,
            override_threshold=thr,
        ).to(device)

        # Make the router sensitive to the spike channels so the effect is visible.
        # In a real system this sensitivity would arise from training or DPO.
        with torch.no_grad():
            model.router.weight.data[0, :4] += 3.0

        stats = evaluate_routing(model, benign, safety)
        print(
            f"threshold={thr:.2f} | "
            f"override benign={stats['override_rate_benign']:.3f}  "
            f"safety={stats['override_rate_safety']:.3f} | "
            f"mean_prot_prob b={stats['mean_protected_prob_benign']:.3f} "
            f"s={stats['mean_protected_prob_safety']:.3f}"
        )

    print("\nAll synthetic checks completed successfully.")
