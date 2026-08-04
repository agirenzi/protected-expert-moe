"""
ProtectedExpertMoE
==================

A minimal, fully vectorized Mixture-of-Experts layer that isolates one
high-precision "protected" expert from the rest of the network.

Design goals
------------
- Router and protected expert stay in a declared high-precision dtype
  (bfloat16 or float32).
- General experts may later be quantized independently.
- Hard override: if the protected expert's routing probability exceeds a
  threshold, 100 % of that token is forced through the protected path.
- Explicit precision assertion so silent quantization is detectable.
- No Python-level token loops.

This is a research prototype intended for controlled experiments on
activation-outlier isolation and post-training quantization robustness.
It is not production MoE infrastructure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class StandardExpert(nn.Module):
    """SwiGLU-style FFN that is allowed to run in lower precision."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ProtectedExpert(nn.Module):
    """
    High-precision expert. All internal computation is forced into the
    protected dtype regardless of the incoming tensor dtype.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        protected_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.protected_dtype = protected_dtype
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)
        self.to(protected_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(self.protected_dtype)
        out = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return out.to(orig_dtype)


class ProtectedExpertMoE(nn.Module):
    """
    Minimal MoE layer with one protected expert and a hard override.

    Parameters
    ----------
    d_model : int
        Residual stream dimension.
    d_hidden : int
        Intermediate size of each expert.
    num_general_experts : int
        Number of low-precision experts.
    protected_dtype : torch.dtype
        Dtype forced on the router and the protected expert.
    override_threshold : float
        If the protected expert's routing probability is >= this value,
        the token is forced entirely through the protected expert.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_hidden: int = 1024,
        num_general_experts: int = 3,
        protected_dtype: torch.dtype = torch.bfloat16,
        override_threshold: float = 0.55,
    ):
        super().__init__()
        if num_general_experts < 1:
            raise ValueError("num_general_experts must be >= 1")

        self.d_model = d_model
        self.num_general_experts = num_general_experts
        self.protected_dtype = protected_dtype
        self.override_threshold = float(override_threshold)

        self.num_experts = num_general_experts + 1
        self.protected_idx = 0

        # Router stays high precision
        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        self.router.to(protected_dtype)

        # Protected expert
        self.protected_expert = ProtectedExpert(
            d_model, d_hidden, protected_dtype
        )

        # General experts
        self.general_experts = nn.ModuleList(
            [StandardExpert(d_model, d_hidden) for _ in range(num_general_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Parameters
        ----------
        x : Tensor
            Shape [batch, seq, d_model] or [N, d_model].
        return_router_info : bool
            If True, also return a dictionary of routing diagnostics.

        Returns
        -------
        out : Tensor
            Same shape as x.
        info : dict (optional)
            probs, override_mask, protected_prob, top1_idx.
        """
        original_shape = x.shape
        if x.dim() == 3:
            b, s, d = x.shape
            flat = x.reshape(b * s, d)
        elif x.dim() == 2:
            flat = x
            b = s = None
        else:
            raise ValueError(f"Expected 2-D or 3-D input, got {x.dim()}-D")

        # High-precision routing
        flat_hp = flat.to(self.protected_dtype)
        logits = self.router(flat_hp)
        probs = F.softmax(logits.float(), dim=-1)  # float32 for numerical stability

        protected_prob = probs[:, self.protected_idx]
        override_mask = protected_prob >= self.override_threshold
        top1_prob, top1_idx = probs.max(dim=-1)

        out = torch.zeros_like(flat)

        # Protected path
        need_protected = override_mask | (top1_idx == self.protected_idx)
        if need_protected.any():
            protected_in = flat[need_protected]
            protected_out = self.protected_expert(protected_in)
            scale = torch.where(
                override_mask[need_protected],
                protected_prob[need_protected],
                top1_prob[need_protected],
            ).unsqueeze(-1).to(protected_out.dtype)
            out[need_protected] = protected_out * scale

        # General experts
        for g_idx in range(self.num_general_experts):
            expert_id = g_idx + 1
            mask = (~override_mask) & (top1_idx == expert_id)
            if mask.any():
                expert_out = self.general_experts[g_idx](flat[mask])
                scale = top1_prob[mask].unsqueeze(-1).to(expert_out.dtype)
                out[mask] = expert_out * scale

        if x.dim() == 3:
            out = out.view(b, s, d)

        if return_router_info:
            info = {
                "probs": probs.detach(),
                "override_mask": override_mask.detach(),
                "protected_prob": protected_prob.detach(),
                "top1_idx": top1_idx.detach(),
            }
            return out, info
        return out

    def assert_protected_precision(self) -> None:
        """Raise if any protected parameter has been moved out of the declared dtype."""
        for name, p in self.named_parameters():
            if "protected_expert" in name or name.startswith("router"):
                if p.dtype != self.protected_dtype:
                    raise RuntimeError(
                        f"Protected parameter '{name}' has dtype {p.dtype}, "
                        f"expected {self.protected_dtype}"
                    )


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float32

    model = ProtectedExpertMoE(
        d_model=128,
        d_hidden=256,
        num_general_experts=2,
        protected_dtype=dtype,
        override_threshold=0.55,
    ).to(device)

    model.assert_protected_precision()
    x = torch.randn(4, 16, 128, device=device, dtype=torch.float32)
    y, info = model(x, return_router_info=True)

    print("Output shape          :", tuple(y.shape))
    print("Override fraction     :", f"{info['override_mask'].float().mean().item():.4f}")
    print("Mean protected prob   :", f"{info['protected_prob'].mean().item():.4f}")
    print("Precision check       : passed")
    print("Forward pass          : succeeded")
