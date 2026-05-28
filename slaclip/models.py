from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CIFARConvNetAvgPoolGAP(nn.Sequential):

    def __init__(self, num_classes: int = 10):
        super().__init__(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128, num_classes, bias=True),
        )


class LeNetCNN(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class IMDBDeepAveragingMLP(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


def _get_dp_rnn_layer(rnn_arch: str):
    rnn_arch = str(rnn_arch).lower().strip()
    try:
        from opacus.layers import DPLSTM, DPGRU
        if rnn_arch == "gru":
            return DPGRU
        return DPLSTM
    except Exception:
        if rnn_arch == "gru":
            return nn.GRU
        return nn.LSTM


class NamesCharDPLSTM(nn.Module):

    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        hidden_size: int = 128,
        n_layers: int = 1,
        embedding_dim: int = 128,
        dropout: float = 0.0,
        pad_id: int = 0,
        rnn_arch: str = "lstm",
        bidirectional: bool = False,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.bidirectional = bool(bidirectional)
        self.rnn_arch = str(rnn_arch).lower().strip()

        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=self.pad_id)
        rnn_type = _get_dp_rnn_layer(self.rnn_arch)
        rnn_dropout = float(dropout) if int(n_layers) > 1 else 0.0
        self.rnn = rnn_type(
            int(embedding_dim),
            int(hidden_size),
            num_layers=int(n_layers),
            dropout=rnn_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        out_dim = int(hidden_size) * (2 if self.bidirectional else 1)
        self.out_layer = nn.Linear(out_dim, int(num_classes))

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        if x.dim() != 2:
            raise ValueError(f"Expected x with shape [B,T], got {tuple(x.shape)}")
        with torch.no_grad():
            lengths = (x != self.pad_id).long().sum(dim=1).clamp(min=1)
        emb = self.embedding(x)
        out, _h = self.rnn(emb, hidden)
        idx = (lengths - 1).view(-1, 1, 1).to(out.device)
        idx = idx.expand(-1, 1, out.size(-1))
        last = out.gather(dim=1, index=idx).squeeze(1)
        logits = self.out_layer(last)
        return logits


def make_model(dataset: str, num_classes: int, meta: dict, args) -> nn.Module:
    ds = dataset.lower().strip()
    if ds == "cifar10":
        return CIFARConvNetAvgPoolGAP(num_classes=num_classes)
    if ds in {"mnist", "fmnist"}:
        return LeNetCNN(num_classes=num_classes)
    if ds == "imdb":
        return IMDBDeepAveragingMLP(vocab_size=int(meta["vocab_size"]))
    if ds == "names":
        return NamesCharDPLSTM(
            vocab_size=int(meta["vocab_size"]),
            num_classes=int(meta["num_classes"]),
            hidden_size=int(args.hidden_size),
            n_layers=int(args.n_layers),
            embedding_dim=int(args.embedding_dim),
            dropout=float(args.dropout),
            pad_id=int(meta["pad_id"]),
            rnn_arch=str(args.rnn_arch),
            bidirectional=bool(args.bidirectional),
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
