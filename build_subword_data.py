# -*- coding: utf-8 -*-
"""
Train two BPE tokenizers (UR source, Roman target) on data/parallel_ur_roman.tsv
and produce:
- data/bpe_src.json, data/bpe_tgt.json
- data/{train,val,test}.pt  (dict with 'src', 'tgt', 'src_len', 'tgt_len')
- a tiny Dataset class for later use.

Usage examples:
  python3 build_subword_data.py --tsv data/parallel_ur_roman.tsv --src_vocab 8000 --tgt_vocab 8000
  python3 build_subword_data.py --encode_only  # re-encode with existing JSON tokenizers
"""
from __future__ import annotations
import argparse, random, json, torch
from pathlib import Path
from typing import List, Tuple
from bpe_tokenizer import BPETokenizer, PAD, BOS, EOS

random.seed(42)

def read_tsv(tsv_path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ur, ro = line.rstrip("\n").split("\t")
                pairs.append((ur, ro))
            except ValueError:
                continue
    return pairs

def split_pairs(pairs: List[Tuple[str, str]], val_ratio=0.05, test_ratio=0.05):
    random.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = pairs[:n_test]
    val = pairs[n_test:n_test+n_val]
    train = pairs[n_test+n_val:]
    return train, val, test

def train_tokenizer(lines: List[str], vocab_size: int, min_freq: int = 2) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(lines, vocab_size=vocab_size, min_freq=min_freq)
    return tok

def encode_examples(tok_src: BPETokenizer, tok_tgt: BPETokenizer, pairs: List[Tuple[str,str]]):
    src_ids, tgt_ids, src_len, tgt_len = [], [], [], []
    pad_id = tok_src.vocab[PAD]
    for ur, ro in pairs:
        s = tok_src.encode(ur, add_bos_eos=True)
        t = tok_tgt.encode(ro, add_bos_eos=True)
        src_ids.append(torch.tensor(s, dtype=torch.long))
        tgt_ids.append(torch.tensor(t, dtype=torch.long))
        src_len.append(len(s))
        tgt_len.append(len(t))
    return src_ids, tgt_ids, torch.tensor(src_len), torch.tensor(tgt_len), pad_id

def pad_batch(seqs, pad_id):
    maxlen = max(len(x) for x in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out

def save_split(split_name: str, outdir: Path, src_ids, tgt_ids, src_len, tgt_len, pad_src):
    X = pad_batch(src_ids, pad_src)
    Y = pad_batch(tgt_ids, pad_src)  # pad id can differ for tgt, but we only need masking later
    obj = {"src": X, "tgt": Y, "src_len": src_len, "tgt_len": tgt_len}
    torch.save(obj, outdir / f"{split_name}.pt")

class Seq2SeqParallelDataset(torch.utils.data.Dataset):
    """
    Later usage:
      ds = Seq2SeqParallelDataset(torch.load("data/train.pt"))
      item = ds[0]  -> dict('src','tgt','src_len','tgt_len')
    """
    def __init__(self, blob: dict):
        self.src = blob["src"]
        self.tgt = blob["tgt"]
        self.src_len = blob["src_len"]
        self.tgt_len = blob["tgt_len"]

    def __len__(self): return self.src.size(0)

    def __getitem__(self, idx):
        return {
            "src": self.src[idx],
            "tgt": self.tgt[idx],
            "src_len": int(self.src_len[idx]),
            "tgt_len": int(self.tgt_len[idx]),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=str, default="data/parallel_ur_roman.tsv")
    ap.add_argument("--src_vocab", type=int, default=8000)
    ap.add_argument("--tgt_vocab", type=int, default=8000)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument("--encode_only", action="store_true",
                    help="Use existing tokenizers at data/bpe_src.json & data/bpe_tgt.json")
    args = ap.parse_args()

    tsv_path = Path(args.tsv)
    outdir = Path("data")
    outdir.mkdir(exist_ok=True, parents=True)

    pairs = read_tsv(tsv_path)
    train, val, test = split_pairs(pairs, args.val_ratio, args.test_ratio)

    if args.encode_only:
        tok_src = BPETokenizer.load(outdir / "bpe_src.json")
        tok_tgt = BPETokenizer.load(outdir / "bpe_tgt.json")
    else:
        # Train on TRAIN ONLY to avoid val/test leakage
        tok_src = train_tokenizer([u for u,_ in train], vocab_size=args.src_vocab, min_freq=args.min_freq)
        tok_tgt = train_tokenizer([r for _,r in train], vocab_size=args.tgt_vocab, min_freq=args.min_freq)
        tok_src.save(outdir / "bpe_src.json")
        tok_tgt.save(outdir / "bpe_tgt.json")

    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        src_ids, tgt_ids, src_len, tgt_len, pad_src = encode_examples(tok_src, tok_tgt, data)
        save_split(split_name, outdir, src_ids, tgt_ids, src_len, tgt_len, pad_src)

    # dump quick stats
    stats = {
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "src_vocab": len(tok_src.id2tok),
        "tgt_vocab": len(tok_tgt.id2tok),
    }
    with open(outdir / "bpe_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Done.\nStats:", stats)

if __name__ == "__main__":
    main()
