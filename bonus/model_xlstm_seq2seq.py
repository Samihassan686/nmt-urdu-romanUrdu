# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

# ---- xLSTM (sLSTM-like) cell with exponential gating ----
class xLSTMCell(nn.Module):
    """
    Minimal sLSTM-style cell with exponential gates (sigma_e = exp(normalized preact) clipped),
    plus layer-norm for stability. This is a pragmatic, compact variant tailored for coursework.
    References: Beck et al., 2024 (xLSTM).
    """
    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.ln = nn.LayerNorm(4 * hidden_size) if layer_norm else nn.Identity()

        # output projection (optional residual out)
        self.out_ln = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()

    @staticmethod
    def exp_gate(x: torch.Tensor) -> torch.Tensor:
        # Exponential gating as per paper; clip to avoid overflow
        return torch.clamp(torch.exp(x), min=0.0, max=20.0)

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h, c = state  # (B,H)
        z = self.ln(self.W(x_t) + self.U(h))  # (B,4H)
        i, f, o, g = torch.chunk(z, 4, dim=-1)

        # exponential gates
        i = self.exp_gate(i)
        f = self.exp_gate(f)
        o = torch.sigmoid(o)           # keep o in [0,1] for stability
        g = torch.tanh(g)

        # scalar memory mixing (sLSTM-style): c' = f ⊙ c + i ⊙ g
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        h_new = self.out_ln(h_new)
        return h_new, c_new

class xLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.cell = xLSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # x: (B,T,In)
        B, T, _ = x.size()
        hs = []
        h_t, c_t = h, c
        for t in range(T):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            hs.append(h_t)
        H = torch.stack(hs, dim=1)  # (B,T,H)
        return self.dropout(H), h_t, c_t

class xLSTMStack(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_dim = input_size if l == 0 else hidden_size
            self.layers.append(xLSTMLayer(in_dim, hidden_size, dropout))

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None, c0: torch.Tensor = None):
        # x: (B,T,In)
        B, T, _ = x.size()
        Hs = []
        h_last, c_last = [], []
        for l, layer in enumerate(self.layers):
            h_init = torch.zeros(B, layer.cell.hidden_size, device=x.device) if h0 is None else h0[l]
            c_init = torch.zeros(B, layer.cell.hidden_size, device=x.device) if c0 is None else c0[l]
            x, h_t, c_t = layer(x, h_init, c_init)  # x becomes (B,T,H)
            Hs.append(x)
            h_last.append(h_t)
            c_last.append(c_t)
        h_last = torch.stack(h_last, dim=0)  # (L,B,H)
        c_last = torch.stack(c_last, dim=0)  # (L,B,H)
        return x, (h_last, c_last)

# ---- Seq2Seq built with xLSTM stacks (encoder & decoder) ----
class xEncoder(nn.Module):
    def __init__(self, vocab: int, emb: int, hidden: int, layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.stack = xLSTMStack(emb, hidden, layers, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor):
        # For simplicity we do not pack here; xLSTMLayer iterates tokens.
        emb = self.dropout(self.embedding(src))  # (B,T,E)
        outs, (h, c) = self.stack(emb)
        return outs, (h, c)

class xDecoder(nn.Module):
    def __init__(self, vocab: int, emb: int, hidden: int, layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.stack = xLSTMStack(emb, hidden, layers, dropout)
        self.proj = nn.Linear(hidden, vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_inp: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
        emb = self.dropout(self.embedding(tgt_inp))
        outs, (h, c) = self.stack(emb, h0, c0)
        logits = self.proj(outs)
        return logits, (h, c)

class xSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 src_pad, tgt_pad,
                 emb_src=256, emb_tgt=256,
                 enc_hidden=512, dec_hidden=512,
                 enc_layers=2, dec_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = xEncoder(src_vocab, emb_src, enc_hidden, enc_layers, dropout, src_pad)
        self.decoder = xDecoder(tgt_vocab, emb_tgt, dec_hidden, dec_layers, dropout, tgt_pad)
        # bridge encoder top state -> decoder initial state (project per layer)
        self.h_proj = nn.Linear(enc_hidden, dec_hidden * dec_layers)
        self.c_proj = nn.Linear(enc_hidden, dec_hidden * dec_layers)
        self.dec_layers = dec_layers
        self.dec_hidden = dec_hidden

    def forward(self, src, src_len, tgt_inp):
        enc_outs, (h_n, c_n) = self.encoder(src, src_len)
        # take top layer last state: (B,H)
        h_top = h_n[-1]
        c_top = c_n[-1]
        B = src.size(0)
        h0 = self.h_proj(h_top).view(B, self.dec_layers, self.dec_hidden).transpose(0,1).contiguous()
        c0 = self.c_proj(c_top).view(B, self.dec_layers, self.dec_hidden).transpose(0,1).contiguous()
        logits, _ = self.decoder(tgt_inp, h0, c0)
        return logits
