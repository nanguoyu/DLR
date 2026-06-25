"""Model wrapper for applying DLR to existing PyTorch modules."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

import torch.nn as nn

from .layers import DLRLinear


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class DLRConfig:
    rank: int
    target_modules: List[str]
    low_rank_alpha: Optional[float] = None
    trainable_low_rank_scale: bool = False
    activation: str = "silu"
    dlr_alpha: float = 1.0
    trainable_dlr_alpha: bool = False
    variance_correction: bool = True
    dlr_scope: str = "all"
    expand_mode: str = "uniform"
    random_seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class DLRModel(nn.Module):
    """Replace selected ``nn.Linear`` modules with ``DLRLinear``.

    Module selection is substring-based. This keeps the wrapper independent of
    a particular model family while matching standard Hugging Face LLaMA names.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        rank: int,
        target_modules: Optional[Iterable[str]] = None,
        low_rank_alpha: Optional[float] = None,
        trainable_low_rank_scale: bool = False,
        activation: str = "silu",
        dlr_alpha: float = 1.0,
        trainable_dlr_alpha: bool = False,
        variance_correction: bool = True,
        dlr_scope: str = "all",
        expand_mode: str = "uniform",
        random_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        targets = list(DEFAULT_TARGET_MODULES if target_modules is None else target_modules)
        self.wrapped_model = model
        self.dlr_config = DLRConfig(
            rank=rank,
            target_modules=targets,
            low_rank_alpha=low_rank_alpha,
            trainable_low_rank_scale=trainable_low_rank_scale,
            activation=activation,
            dlr_alpha=dlr_alpha,
            trainable_dlr_alpha=trainable_dlr_alpha,
            variance_correction=variance_correction,
            dlr_scope=dlr_scope,
            expand_mode=expand_mode,
            random_seed=random_seed,
        )
        if hasattr(model, "config"):
            self.config = model.config

        replacements = []
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(key in module_name for key in targets):
                replacements.append((module_name, module))

        for module_name, module in replacements:
            use_dlr = self._scope_matches(module_name, dlr_scope)
            new_module = DLRLinear(
                module.in_features,
                module.out_features,
                rank,
                low_rank_alpha=low_rank_alpha,
                trainable_low_rank_scale=trainable_low_rank_scale,
                activation=activation,
                dlr_alpha=dlr_alpha,
                trainable_dlr_alpha=trainable_dlr_alpha,
                variance_correction=variance_correction,
                use_dlr=use_dlr,
                expand_mode=expand_mode,
                random_seed=random_seed,
                module_name=module_name,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            parent = self._parent(module_name)
            setattr(parent, module_name.rsplit(".", 1)[-1], new_module)
            if verbose:
                status = "DLR" if use_dlr else "low-rank"
                print(f"[{status}] replaced {module_name}")

    @staticmethod
    def _scope_matches(module_name: str, scope: str) -> bool:
        scope = scope.lower()
        name = module_name.lower()
        if scope == "all":
            return True
        if scope == "none":
            return False
        if scope == "mlp":
            return "mlp" in name
        if scope == "attn":
            return "attn" in name or "attention" in name
        return scope in name

    def _parent(self, module_name: str) -> nn.Module:
        parent = self.wrapped_model
        for part in module_name.split(".")[:-1]:
            parent = getattr(parent, part)
        return parent

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_model, name)

    def dlr_layers(self):
        for module in self.wrapped_model.modules():
            if isinstance(module, DLRLinear):
                yield module

    def fold(self) -> int:
        """Fold all active DLR branches and return the number folded."""

        n_folded = 0
        for module in self.dlr_layers():
            n_folded += int(module.fold())
        return n_folded
