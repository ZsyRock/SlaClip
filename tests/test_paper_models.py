import sys
from pathlib import Path

import torch


_SLACLIP_DIR = Path(__file__).resolve().parents[1]
_PATCHES_DIR = _SLACLIP_DIR / "patches"
_OPACUS_ROOT = _SLACLIP_DIR.parent
sys.path.insert(0, str(_PATCHES_DIR))
sys.path.insert(1, str(_SLACLIP_DIR))
sys.path.insert(2, str(_OPACUS_ROOT))

from slaclip.models import IMDBDeepAveragingMLP, NamesCharDPLSTM


def test_imdb_average_ignores_right_padding():
    torch.manual_seed(0)
    model = IMDBDeepAveragingMLP(vocab_size=32, pad_id=0).eval()
    short = torch.tensor([[4, 7, 9]], dtype=torch.long)
    padded = torch.tensor([[4, 7, 9, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        short_logits = model(short)
        padded_logits = model(padded)

    assert torch.allclose(short_logits, padded_logits, atol=1e-7, rtol=0.0)


def test_names_model_uses_opacus_dp_lstm():
    model = NamesCharDPLSTM(
        vocab_size=16,
        num_classes=3,
        hidden_size=8,
        embedding_dim=8,
        rnn_arch="lstm",
    )
    assert model.rnn.__class__.__name__ == "DPLSTM"


def test_names_last_valid_timestep_ignores_right_padding():
    torch.manual_seed(0)
    model = NamesCharDPLSTM(
        vocab_size=16,
        num_classes=3,
        hidden_size=8,
        embedding_dim=8,
        pad_id=0,
        rnn_arch="lstm",
    ).eval()
    short = torch.tensor([[2, 4, 5]], dtype=torch.long)
    padded = torch.tensor([[2, 4, 5, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        short_logits = model(short)
        padded_logits = model(padded)

    assert torch.allclose(short_logits, padded_logits, atol=1e-7, rtol=0.0)
