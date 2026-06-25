"""Core DLR layers.

DLR augments a low-rank decoder with a fixed latent residual during training:

    y = B z + alpha / sqrt(K) * Expand_K(z) + b.

Since Expand_K is linear in z, the residual can be folded into B after
training, leaving no inference-time DLR branch.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _stable_seed(base_seed: int, name: str) -> int:
    name_hash = int.from_bytes(
        hashlib.md5(name.encode("utf-8", errors="ignore")).digest()[:8],
        "little",
        signed=False,
    )
    return int((int(base_seed) ^ name_hash) % (2**63 - 1))


def expansion_index(
    out_features: int,
    rank: int,
    *,
    mode: str = "uniform",
    random_seed: int = 0,
    module_name: str = "",
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return the output-to-latent index used by Expand_K.

    The returned tensor has shape ``[out_features]``. Entry ``idx[o]`` gives the
    latent coordinate copied into output coordinate ``o``.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")

    mode = mode.lower()
    if mode == "uniform":
        k = math.ceil(out_features / rank)
        idx = torch.arange(out_features, device=device, dtype=torch.long) // k
        return idx.clamp_max(rank - 1)
    if mode == "random_reuse":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_stable_seed(random_seed, module_name))
        idx = torch.randint(0, rank, (out_features,), generator=gen, dtype=torch.long)
        return idx.to(device=device)
    raise ValueError(f"unsupported expansion mode: {mode!r}")


def expand_latent(
    z: Tensor,
    out_features: int,
    *,
    mode: str = "uniform",
    random_index: Optional[Tensor] = None,
) -> Tensor:
    """Expand latent states from ``[..., rank]`` to ``[..., out_features]``."""

    if z.shape[-1] <= 0:
        raise ValueError("latent rank must be positive")

    mode = mode.lower()
    if mode == "uniform":
        k = math.ceil(out_features / z.shape[-1])
        expanded = z.repeat_interleave(k, dim=-1)
        return expanded[..., :out_features]
    if mode == "random_reuse":
        if random_index is None:
            raise ValueError("random_index is required for random_reuse expansion")
        return z.index_select(dim=-1, index=random_index.to(device=z.device, dtype=torch.long))
    raise ValueError(f"unsupported expansion mode: {mode!r}")


def _activation(name: str):
    name = name.lower()
    if name in ("identity", "none", "linear"):
        return None
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"unsupported activation: {name!r}")


class DLRLinear(nn.Module):
    """Low-rank linear layer with a foldable DLR residual.

    Args:
        in_features: input dimension.
        out_features: output dimension.
        rank: low-rank latent dimension ``r``.
        low_rank_alpha: scale applied to the low-rank latent; defaults to
            ``rank`` so the effective multiplier ``low_rank_alpha / rank`` is 1.
        trainable_low_rank_scale: if true, uses a trainable scalar for the
            low-rank latent scale.
        activation: activation between down- and up-projections.
        dlr_alpha: DLR residual strength.
        trainable_dlr_alpha: if true, makes DLR alpha trainable. The paper uses
            a fixed alpha by default.
        variance_correction: scale DLR by ``1 / sqrt(K)``.
        use_dlr: if false, this is a plain low-rank layer.
        expand_mode: ``"uniform"`` for contiguous duplication or
            ``"random_reuse"`` for the fixed random ablation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        low_rank_alpha: Optional[float] = None,
        trainable_low_rank_scale: bool = False,
        activation: str = "silu",
        dlr_alpha: float = 1.0,
        trainable_dlr_alpha: bool = False,
        variance_correction: bool = True,
        use_dlr: bool = True,
        expand_mode: str = "uniform",
        random_seed: int = 0,
        module_name: str = "",
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.k = math.ceil(self.out_features / self.rank)
        self.use_dlr = bool(use_dlr)
        self.variance_correction = bool(variance_correction)
        self.expand_mode = expand_mode.lower()
        self.module_name = module_name
        self._dlr_folded = False

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        low_rank_scale = 1.0 if low_rank_alpha is None else float(low_rank_alpha) / float(rank)
        if trainable_low_rank_scale:
            self.low_rank_scale = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
            self._fixed_low_rank_scale = None
        else:
            self.register_buffer(
                "low_rank_scale",
                torch.tensor(low_rank_scale, device=device, dtype=dtype),
            )
            self._fixed_low_rank_scale = low_rank_scale

        if trainable_dlr_alpha:
            self.dlr_alpha = nn.Parameter(torch.tensor(float(dlr_alpha), device=device, dtype=dtype))
        else:
            self.register_buffer("dlr_alpha", torch.tensor(float(dlr_alpha), device=device, dtype=dtype))

        if self.expand_mode == "random_reuse":
            idx = expansion_index(
                out_features,
                rank,
                mode="random_reuse",
                random_seed=random_seed,
                module_name=module_name,
                device=torch.device("cpu"),
            )
            self.register_buffer("random_index", idx, persistent=True)
        elif self.expand_mode == "uniform":
            self.register_buffer("random_index", None, persistent=False)
        else:
            raise ValueError(f"unsupported expansion mode: {expand_mode!r}")

        self.act = _activation(activation)
        self.activation_name = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        target_sdv = (self.in_features + self.out_features) ** -0.5
        init_scale = (self.rank ** -0.25) * (target_sdv ** 0.5)
        with torch.no_grad():
            self.down.weight.normal_(mean=0.0, std=init_scale)
            self.up.weight.normal_(mean=0.0, std=init_scale)
            if self.bias is not None:
                bound = 1.0 / math.sqrt(self.out_features)
                self.bias.uniform_(-bound, bound)

    def _latent(self, x: Tensor) -> Tensor:
        z = self.down(x)
        if self.act is not None:
            z = self.act(z)
        scale = self.low_rank_scale.tanh() if isinstance(self.low_rank_scale, nn.Parameter) else self.low_rank_scale
        return z * scale

    def _dlr_scale(self) -> Tensor:
        if self.variance_correction:
            return self.dlr_alpha / math.sqrt(self.k)
        return self.dlr_alpha

    def _expand_idx(self) -> Tensor:
        return expansion_index(
            self.out_features,
            self.rank,
            mode=self.expand_mode,
            random_seed=0,
            module_name=self.module_name,
            device=self.up.weight.device,
        ) if self.expand_mode == "uniform" else self.random_index.to(self.up.weight.device)

    def forward(self, x: Tensor) -> Tensor:
        z = self._latent(x)
        out = self.up(z)
        if self.use_dlr:
            out = out + expand_latent(
                z,
                self.out_features,
                mode=self.expand_mode,
                random_index=self.random_index,
            ) * self._dlr_scale()
        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.no_grad()
    def fold(self) -> bool:
        """Fold the DLR residual into ``self.up.weight``.

        Returns ``True`` if this call changed the layer and ``False`` if it was
        already folded or had no active DLR branch.
        """

        if self._dlr_folded or not self.use_dlr:
            self._dlr_folded = True
            return False

        idx = self._expand_idx()
        r_matrix = F.one_hot(idx, num_classes=self.rank).to(
            dtype=self.up.weight.dtype,
            device=self.up.weight.device,
        )
        self.up.weight.add_(r_matrix, alpha=float(self._dlr_scale().detach().cpu()))
        self.use_dlr = False
        self._dlr_folded = True
        return True

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, k={self.k}, activation={self.activation_name}, "
            f"use_dlr={self.use_dlr}, expand_mode={self.expand_mode}"
        )
