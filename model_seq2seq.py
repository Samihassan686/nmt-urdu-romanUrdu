# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor):
        # src: (B, Tsrc)
        emb = self.dropout(self.embedding(src))  # (B, Tsrc, E)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (h_n, c_n) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # (B, Tsrc, 2H)
        # h_n: (L*2, B, H), c_n: (L*2, B, H)
        return outputs, (h_n, c_n)

class BridgeToDecoderInit(nn.Module):
    """
    Maps the top encoder layer's bi-directional states into initial states for a multi-layer decoder.
    Produces (h0, c0) of shape (L_dec, B, H_dec).
    """
    def __init__(self, enc_hidden: int, dec_hidden: int, dec_layers: int):
        super().__init__()
        self.enc_cat_size = enc_hidden * 2
        self.dec_layers = dec_layers
        self.dec_hidden = dec_hidden
        self.h_proj = nn.Linear(self.enc_cat_size, dec_layers * dec_hidden)
        self.c_proj = nn.Linear(self.enc_cat_size, dec_layers * dec_hidden)

    def forward(self, enc_hn: torch.Tensor, enc_cn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # enc_hn, enc_cn: (L_enc*2, B, H_enc)
        # Take the top encoder layer (last two slices: forward & backward), concat on hidden dim
        B = enc_hn.size(1)
        h_top_f = enc_hn[-2]  # (B, H_enc)
        h_top_b = enc_hn[-1]  # (B, H_enc)
        c_top_f = enc_cn[-2]
        c_top_b = enc_cn[-1]
        h_cat = torch.cat([h_top_f, h_top_b], dim=1)  # (B, 2H_enc)
        c_cat = torch.cat([c_top_f, c_top_b], dim=1)  # (B, 2H_enc)

        h0 = self.h_proj(h_cat).view(B, self.dec_layers, self.dec_hidden).transpose(0, 1).contiguous()
        c0 = self.c_proj(c_cat).view(B, self.dec_layers, self.dec_hidden).transpose(0, 1).contiguous()
        # shapes: (L_dec, B, H_dec)
        return h0, c0

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int = 4, dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_inp: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
        emb = self.dropout(self.embedding(tgt_inp))  # (B, Ttgt, E)
        out, (h_n, c_n) = self.rnn(emb, (h0, c0))    # out: (B, Ttgt, H)
        logits = self.proj(out)                       # (B, Ttgt, V)
        return logits, (h_n, c_n)

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int,
                 src_pad: int, tgt_pad: int,
                 emb_src: int = 256, emb_tgt: int = 256,
                 enc_hidden: int = 512, dec_hidden: int = 512,
                 enc_layers: int = 2, dec_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_src, enc_hidden, enc_layers, dropout, src_pad)
        self.bridge = BridgeToDecoderInit(enc_hidden, dec_hidden, dec_layers)
        self.decoder = Decoder(tgt_vocab, emb_tgt, dec_hidden, dec_layers, dropout, tgt_pad)

    def forward(self, src, src_len, tgt_inp):
        enc_outs, (h_n, c_n) = self.encoder(src, src_len)
        h0, c0 = self.bridge(h_n, c_n)
        logits, _ = self.decoder(tgt_inp, h0, c0)
        return logits  # (B, Ttgt, V)
