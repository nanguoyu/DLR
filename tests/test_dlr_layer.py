import torch
import torch.nn as nn

from dlr import DLRLinear, DLRModel, expand_latent, expansion_index


def test_uniform_expand_matches_repeat_interleave():
    z = torch.arange(12, dtype=torch.float32).view(2, 6)
    out = expand_latent(z, out_features=14)
    expected = z.repeat_interleave(3, dim=-1)[:, :14]
    assert torch.equal(out, expected)


def test_expansion_index_uniform():
    idx = expansion_index(10, 4)
    assert idx.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]


def test_dlr_linear_fold_equivalence():
    torch.manual_seed(0)
    layer = DLRLinear(16, 31, rank=7, activation="silu", dlr_alpha=1.0)
    layer.eval()
    x = torch.randn(5, 3, 16)

    with torch.no_grad():
        y_before = layer(x)
        changed = layer.fold()
        y_after = layer(x)

    assert changed is True
    assert layer.use_dlr is False
    assert torch.allclose(y_before, y_after, rtol=1e-5, atol=1e-6)
    assert layer.fold() is False


def test_dlr_model_replacement_and_fold():
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.other = nn.Linear(8, 8)

        def forward(self, x):
            return self.other(self.q_proj(x))

    model = DLRModel(TinyBlock(), target_modules=["q_proj"], rank=4, activation="identity")
    assert isinstance(model.wrapped_model.q_proj, DLRLinear)
    assert isinstance(model.wrapped_model.other, nn.Linear)

    x = torch.randn(2, 8)
    model.eval()
    with torch.no_grad():
        y_before = model(x)
        assert model.fold() == 1
        y_after = model(x)

    assert torch.allclose(y_before, y_after, rtol=1e-5, atol=1e-6)
