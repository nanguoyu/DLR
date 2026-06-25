# DLR: Zero-Inference-Cost Latent Residuals for Low-Rank Pre-Training

This repository contains a compact reference implementation of **Duplicated
Latent Residuals (DLR)**, a training-time residual branch for low-rank
pre-training.

For a low-rank linear layer with latent state `z`, DLR adds

```text
y = B z + alpha / sqrt(K) * Expand_K(z) + b,
K = ceil(d_out / r)
```

where `Expand_K` repeats each latent coordinate across the output dimension and
truncates to `d_out`. After training, the residual is folded into the
up-projection:

```text
B* = B + alpha / sqrt(K) * R
```

The deployed model then uses the same graph, parameter count, FLOPs, and memory
footprint as the underlying low-rank model.

This repository is intentionally small. It includes the DLR layer, a module
wrapper for replacing selected `torch.nn.Linear` layers, fold tests, and a
minimal usage example. It does not include training logs, checkpoints, paper
drafts, cluster scripts, or W&B exports.

## Install

```bash
git clone https://github.com/nanguoyu/DLR.git
cd DLR
pip install -e ".[dev]"
```

For use without tests:

```bash
pip install -e .
```

## Minimal Usage

```python
import torch
from dlr import DLRModel

model = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.SiLU(),
    torch.nn.Linear(64, 32),
)

model = DLRModel(
    model,
    target_modules=["0", "2"],
    rank=8,
    dlr_alpha=1.0,
    activation="silu",
)

x = torch.randn(4, 32)
y_train = model(x)

model.eval()
model.fold()
y_deploy = model(x)
```

`fold()` is exact up to floating-point error and removes the DLR residual branch
from every wrapped layer.

## API

```python
from dlr import DLRLinear, DLRModel, expand_latent
```

`DLRLinear` replaces a dense `Linear(d_in, d_out)` with a rank-`r`
down-projection, a rank-`r` up-projection, and the foldable DLR residual.

`DLRModel` recursively replaces selected `torch.nn.Linear` modules in an
existing model:

```python
wrapped = DLRModel(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    rank=256,
    dlr_alpha=1.0,
    activation="silu",
)
```

The target list is substring-based, matching common Hugging Face LLaMA module
names. Use `dlr_scope="mlp"`, `dlr_scope="attn"`, or `dlr_scope="none"` to
control where the residual branch is active.

## Tests

```bash
python -m pytest tests
```

The tests verify the expansion operator, single-layer fold equivalence, model
wrapper replacement, and idempotent folding.

## Repository Scope

This is the public reference code for the DLR mechanism. The full experimental
training environment used for the paper included cluster-specific launchers and
logging infrastructure; those files are deliberately omitted from this clean
release to avoid exposing local paths, logs, and unrelated baseline code.

## Citation

```bibtex
@misc{wang2026dlr,
  title  = {DLR: Zero-Inference-Cost Latent Residuals for Low-Rank Pre-Training},
  author = {Wang, Dong and Tang, Wenwu and Cheng, Yun and Saukh, Olga},
  year   = {2026},
  note   = {Preprint}
}
```
