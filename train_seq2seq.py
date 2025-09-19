# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math, time, random
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model_seq2seq import Seq2Seq

# Special tokens (must match bpe_tokenizer.py)
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

class Seq2SeqParallelDataset(Dataset):
    """
    Wraps the padded tensors produced by build_subword_data.py
    Each item:
      {
        "src": LongTensor (T_src_max,),
        "tgt": LongTensor (T_tgt_max,),
        "src_len": int,
        "tgt_len": int
      }
    """
    def __init__(self, blob: Dict[str, torch.Tensor]):
        self.src = blob["src"]      # (N, Tsrc_max)
        self.tgt = blob["tgt"]      # (N, Ttgt_max)
        self.src_len = blob["src_len"]  # (N,)
        self.tgt_len = blob["tgt_len"]  # (N,)

    def __len__(self): 
        return self.src.size(0)

    def __getitem__(self, idx):
        return {
            "src": self.src[idx],
            "tgt": self.tgt[idx],
            "src_len": int(self.src_len[idx]),
            "tgt_len": int(self.tgt_len[idx]),
        }

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_pad_idx(tok_json):
    id2tok = tok_json["id2tok"]
    tok2id = {t:i for i,t in enumerate(id2tok)}
    return tok2id[PAD], tok2id[BOS], tok2id[EOS], tok2id[UNK], len(id2tok)

def forcing_inputs(tgt_tensor: torch.Tensor):
    """
    Given padded target sequences with BOS/EOS already inside,
    build (tgt_inp, tgt_gold):
      tgt_inp  = [y_0 .. y_{T-2}]   (includes BOS, drops last token)
      tgt_gold = [y_1 .. y_{T-1}]   (drops BOS)
    Shapes preserved: (B, T-1)
    """
    tgt_inp  = tgt_tensor[:, :-1].contiguous()
    tgt_gold = tgt_tensor[:, 1:].contiguous()
    return tgt_inp, tgt_gold

@torch.no_grad()
def evaluate(model, loader, criterion, pad_id_tgt, device):
    model.eval()
    total_loss, total_tok, total_correct = 0.0, 0, 0
    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_len = torch.tensor(batch["src_len"], dtype=torch.long, device=device)

        tgt_inp, tgt_gold = forcing_inputs(tgt)
        logits = model(src, src_len, tgt_inp)  # (B, T-1, V)
        V = logits.size(-1)
        loss = criterion(logits.view(-1, V), tgt_gold.view(-1))

        # token accuracy (exclude PAD)
        pred = logits.argmax(-1)
        mask = tgt_gold.ne(pad_id_tgt)
        correct = (pred.eq(tgt_gold) & mask).sum().item()
        total_correct += correct
        total_tok += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    avg_nll = total_loss / max(1, total_tok)
    ppl = math.exp(avg_nll)
    acc = total_correct / max(1, total_tok)
    return avg_nll, ppl, acc

def train():
    import argparse
    ap = argparse.ArgumentParser()
    # Data / IO
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--save_name", type=str, default="bilstm_seq2seq.pt")

    # Optim / schedule
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # Model dims / layers
    ap.add_argument("--emb_src", type=int, default=256)
    ap.add_argument("--emb_tgt", type=int, default=256)
    ap.add_argument("--enc_hidden", type=int, default=512)
    ap.add_argument("--dec_hidden", type=int, default=512)
    ap.add_argument("--enc_layers", type=int, default=2)  # BiLSTM encoder layers
    ap.add_argument("--dec_layers", type=int, default=4)  # LSTM decoder layers
    ap.add_argument("--dropout", type=float, default=0.2)

    args = ap.parse_args()

    # Repro
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / args.save_name

    # Load datasets (already split 50/25/25 by build_subword_data.py)
    train_blob = torch.load(data_dir / "train.pt", map_location="cpu")
    val_blob   = torch.load(data_dir / "val.pt", map_location="cpu")
    test_blob  = torch.load(data_dir / "test.pt", map_location="cpu")

    ds_tr = Seq2SeqParallelDataset(train_blob)
    ds_va = Seq2SeqParallelDataset(val_blob)
    ds_te = Seq2SeqParallelDataset(test_blob)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load tokenizers to get vocab sizes & pad indices
    tok_src_json = load_json(data_dir / "bpe_src.json")
    tok_tgt_json = load_json(data_dir / "bpe_tgt.json")
    pad_src, bos_src, eos_src, unk_src, Vsrc = get_pad_idx(tok_src_json)
    pad_tgt, bos_tgt, eos_tgt, unk_tgt, Vtgt = get_pad_idx(tok_tgt_json)

    # Build model
    model = Seq2Seq(
        src_vocab=Vsrc, tgt_vocab=Vtgt,
        src_pad=pad_src, tgt_pad=pad_tgt,
        emb_src=args.emb_src, emb_tgt=args.emb_tgt,
        enc_hidden=args.enc_hidden, dec_hidden=args.dec_hidden,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers,
        dropout=args.dropout
    ).to(device)

    # Loss/optim
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Vsrc={Vsrc}, Vtgt={Vtgt}")
    print(f"Enc: layers={args.enc_layers} hidden={args.enc_hidden} emb={args.emb_src}")
    print(f"Dec: layers={args.dec_layers} hidden={args.dec_hidden} emb={args.emb_tgt}")
    print(f"Dropout={args.dropout}  LR={args.lr}  Batch={args.batch_size}")

    best_val = float("inf")
    patience = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_tok = 0.0, 0
        t0 = time.time()

        for batch in dl_tr:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_len = torch.tensor(batch["src_len"], dtype=torch.long, device=device)

            tgt_inp, tgt_gold = forcing_inputs(tgt)

            logits = model(src, src_len, tgt_inp)  # (B, T-1, V)
            V = logits.size(-1)
            loss = criterion(logits.view(-1, V), tgt_gold.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            # account for token count (exclude pad via mask)
            mask = tgt_gold.ne(pad_tgt)
            epoch_loss += loss.item() * mask.sum().item()
            epoch_tok  += mask.sum().item()

        train_nll = epoch_loss / max(1, epoch_tok)
        train_ppl = math.exp(train_nll)

        val_nll, val_ppl, val_acc = evaluate(model, dl_va, criterion, pad_tgt, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | {dt:.1f}s | train ppl {train_ppl:.2f} | val ppl {val_ppl:.2f} | val acc {val_acc*100:.2f}%")

        # Early stopping on val NLL
        if val_nll < best_val:
            best_val = val_nll
            torch.save({
                "model_state": model.state_dict(),
                "args": {
                    "data_dir": args.data_dir,
                    "emb_src": args.emb_src, "emb_tgt": args.emb_tgt,
                    "enc_hidden": args.enc_hidden, "dec_hidden": args.dec_hidden,
                    "enc_layers": args.enc_layers, "dec_layers": args.dec_layers,
                    "dropout": args.dropout, "lr": args.lr,
                    "batch_size": args.batch_size, "epochs": args.epochs,
                    "clip": args.clip, "patience": args.patience, "seed": args.seed
                },
                "Vsrc": Vsrc, "Vtgt": Vtgt,
                "pad_src": pad_src, "pad_tgt": pad_tgt
            }, ckpt_path)
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_nll, test_ppl, test_acc = evaluate(model, dl_te, criterion, pad_tgt, device)
    print(f"TEST | ppl {test_ppl:.2f} | acc {test_acc*100:.2f}%")

if __name__ == "__main__":
    train()
