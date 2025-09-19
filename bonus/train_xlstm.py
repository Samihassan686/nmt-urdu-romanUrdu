# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math, time, random
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bonus.model_xlstm_seq2seq import xSeq2Seq

PAD = "<pad>"; BOS = "<bos>"; EOS = "<eos>"; UNK = "<unk>"

class Seq2SeqParallelDataset(torch.utils.data.Dataset):
    def __init__(self, blob: Dict[str, torch.Tensor]):
        self.src, self.tgt = blob["src"], blob["tgt"]
        self.src_len, self.tgt_len = blob["src_len"], blob["tgt_len"]
    def __len__(self): return self.src.size(0)
    def __getitem__(self, i):
        return {"src": self.src[i], "tgt": self.tgt[i],
                "src_len": int(self.src_len[i]), "tgt_len": int(self.tgt_len[i])}

def load_json(p): 
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def get_pad_idx(tok_json):
    id2tok = tok_json["id2tok"]; tok2id = {t:i for i,t in enumerate(id2tok)}
    return tok2id[PAD], tok2id[BOS], tok2id[EOS], tok2id[UNK], len(id2tok)

def teacher_forcing_inputs(tgt):
    return tgt[:, :-1].contiguous(), tgt[:, 1:].contiguous()

@torch.no_grad()
def evaluate(model, loader, criterion, pad_id, device):
    model.eval(); total_loss=0.0; total_tok=0; total_correct=0
    for b in loader:
        X=b["src"].to(device); Y=b["tgt"].to(device)
        L=torch.tensor(b["src_len"], device=device).long()
        Y_in, Y_gold = teacher_forcing_inputs(Y)
        logits = model(X, L, Y_in)
        V = logits.size(-1)
        loss = criterion(logits.view(-1,V), Y_gold.view(-1))
        pred = logits.argmax(-1)
        mask = Y_gold.ne(pad_id)
        total_loss += loss.item() * mask.sum().item()
        total_correct += (pred.eq(Y_gold) & mask).sum().item()
        total_tok += mask.sum().item()
    nll = total_loss / max(1,total_tok)
    return nll, math.exp(nll), total_correct/max(1,total_tok)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emb_src", type=int, default=256)
    ap.add_argument("--emb_tgt", type=int, default=256)
    ap.add_argument("--enc_hidden", type=int, default=512)
    ap.add_argument("--dec_hidden", type=int, default=512)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--save_name", type=str, default="xlstm_seq2seq.pt")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_blob = torch.load(Path(args.data_dir)/"train.pt", map_location="cpu")
    val_blob   = torch.load(Path(args.data_dir)/"val.pt", map_location="cpu")
    test_blob  = torch.load(Path(args.data_dir)/"test.pt", map_location="cpu")

    dl_tr = DataLoader(Seq2SeqParallelDataset(train_blob), batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(Seq2SeqParallelDataset(val_blob), batch_size=args.batch_size)
    dl_te = DataLoader(Seq2SeqParallelDataset(test_blob), batch_size=args.batch_size)

    tks = load_json(Path(args.data_dir)/"bpe_src.json")
    tkt = load_json(Path(args.data_dir)/"bpe_tgt.json")
    pad_src, _, _, _, Vsrc = get_pad_idx(tks)
    pad_tgt, _, _, _, Vtgt = get_pad_idx(tkt)

    model = xSeq2Seq(
        src_vocab=Vsrc, tgt_vocab=Vtgt,
        src_pad=pad_src, tgt_pad=pad_tgt,
        emb_src=args.emb_src, emb_tgt=args.emb_tgt,
        enc_hidden=args.enc_hidden, dec_hidden=args.dec_hidden,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        dropout=args.dropout
    ).to(device)

    crit = nn.CrossEntropyLoss(ignore_index=pad_tgt)
    opt  = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf"); patience = args.patience
    ckpt_dir = Path(args.save_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir/args.save_name

    for ep in range(1, args.epochs+1):
        model.train(); t0=time.time(); tok=0; loss_sum=0.0
        for b in dl_tr:
            X=b["src"].to(device); Y=b["tgt"].to(device)
            L=torch.tensor(b["src_len"], device=device).long()
            Y_in, Y_gold = teacher_forcing_inputs(Y)
            logits = model(X, L, Y_in)
            V = logits.size(-1)
            loss = crit(logits.view(-1,V), Y_gold.view(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            mask = Y_gold.ne(pad_tgt)
            loss_sum += loss.item() * mask.sum().item()
            tok += mask.sum().item()
        tr_nll = loss_sum/max(1,tok); tr_ppl = math.exp(tr_nll)
        va_nll, va_ppl, va_acc = evaluate(model, dl_va, crit, pad_tgt, device)
        print(f"Epoch {ep:02d} | train ppl {tr_ppl:.2f} | val ppl {va_ppl:.2f} | val acc {va_acc*100:.2f}%")
        if va_nll < best_val:
            best_val = va_nll
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "Vsrc": Vsrc, "Vtgt": Vtgt, "pad_src": pad_src, "pad_tgt": pad_tgt
            }, ckpt)
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping."); break

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    te_nll, te_ppl, te_acc = evaluate(model, dl_te, crit, pad_tgt, device)
    print(f"TEST | ppl {te_ppl:.2f} | acc {te_acc*100:.2f}%")

if __name__ == "__main__":
    import time, math
    main()
