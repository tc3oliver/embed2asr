"""GRU-CTC reader on frozen EmbeddingGemma 2 temporal audio features (25 frames/s, D=1536).

Architecture is kept byte-compatible with the research checkpoints (state_dict keys and shapes)."""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, k=5, p=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k // 2)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, x, mask):  # x [B,T,D], mask [B,T,1]
        h = self.conv((x * mask).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(torch.nn.functional.gelu(self.norm(h)))


class Reader(nn.Module):
    def __init__(self, in_dim, vocab_size, d=512, gru=256, blocks=3, dropout=0.0, drop_block=None, drop_gru=None, **_):
        super().__init__()
        drop_block = dropout if drop_block is None else drop_block
        drop_gru = dropout if drop_gru is None else drop_gru
        self.cfg = dict(
            in_dim=in_dim, vocab_size=vocab_size, d=d, gru=gru, blocks=blocks, dropout=dropout, drop_block=drop_block, drop_gru=drop_gru
        )
        self.pre = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU())
        self.blocks = nn.ModuleList(Block(d, p=drop_block) for _ in range(blocks))
        # ROCm/MIOpen bug: nn.GRU(num_layers>1, dropout>0) corrupts outputs in train mode (see docs/limitations.md),
        # so inter-layer dropout is done explicitly between two single-layer GRUs.
        self.drop_gru = drop_gru
        if drop_gru > 0:
            self.rnn1 = nn.GRU(d, gru, batch_first=True, bidirectional=True)
            self.rnn2 = nn.GRU(2 * gru, gru, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.GRU(d, gru, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * gru, vocab_size)

    def forward(self, x, lengths):
        mask = (torch.arange(x.size(1), device=x.device)[None] < lengths[:, None].to(x.device)).unsqueeze(-1).to(x.dtype)
        h = self.pre(x)
        for b in self.blocks:
            h = b(h, mask)
        packed = nn.utils.rnn.pack_padded_sequence(h * mask, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if self.drop_gru > 0:
            h, _ = self.rnn1(packed)
            h = nn.utils.rnn.PackedSequence(
                torch.nn.functional.dropout(h.data, self.drop_gru, self.training), h.batch_sizes, h.sorted_indices, h.unsorted_indices
            )
            h, _ = self.rnn2(h)
        else:
            h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        return self.out(h)
