import torch
import torch.nn as nn

if __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlr import DLRModel


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main():
    torch.manual_seed(0)
    model = DLRModel(
        TinyMLP(),
        target_modules=["fc1", "fc2"],
        rank=8,
        dlr_alpha=1.0,
        activation="silu",
    )
    model.eval()

    x = torch.randn(4, 32)
    with torch.no_grad():
        y_before = model(x)
        n_folded = model.fold()
        y_after = model(x)

    max_error = (y_before - y_after).abs().max().item()
    print(f"folded_layers={n_folded}")
    print(f"max_abs_error={max_error:.3e}")


if __name__ == "__main__":
    main()
