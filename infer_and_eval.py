# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math, argparse
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_seq2seq import Seq2Seq
from bpe_tokenizer import BPETokenizer, PAD, BOS, EOS

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def ids_to_masked(logits, pad_id):
    # logits: (B,T,V) -> pred ids (B,T)
    return logits.argmax(-1)

def greedy_decode(model: nn.Module, src, src_len, max_len: int, bos_id: int, eos_id: int, pad_id: int, device):
    model.eval()
    with torch.no_grad():
        # run encoder to get initial state
        enc_outs, (h_n, c_n) = model.encoder(src, src_len)
        h0, c0 = model.bridge(h_n, c_n)
        B = src.size(0)
        ys = torch.full((B,1), bos_id, dtype=torch.long, device=device)
        h, c = h0, c0
        outs = []
        for _ in range(max_len):
            emb = model.decoder.embedding(ys[:, -1:])  # (B,1,E)
            out, (h, c) = model.decoder.rnn(emb, (h, c))
            logit = model.decoder.proj(out[:, -1])  # (B,V)
            next_id = logit.argmax(-1).unsqueeze(1)  # (B,1)
            outs.append(next_id)
            ys = torch.cat([ys, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        pred = torch.cat(outs, dim=1) if outs else torch.empty((B,0), dtype=torch.long, device=device)
        return pred  # without the initial BOS

def bleu_corpus(refs: List[List[str]], hyps: List[List[str]], max_n: int = 4) -> float:
    # Simple corpus BLEU with brevity penalty (no smoothing)
    import collections
    def ngrams(seq, n): return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    weights = [1.0/max_n]*max_n
    clip_counts = [0]*max_n
    total_counts = [0]*max_n
    ref_len = 0
    hyp_len = 0
    for ref, hyp in zip(refs, hyps):
        ref_len += len(ref)
        hyp_len += len(hyp)
        ref_ngrams = [collections.Counter(ngrams(ref, n)) for n in range(1, max_n+1)]
        hyp_ngrams = [collections.Counter(ngrams(hyp, n)) for n in range(1, max_n+1)]
        for i in range(max_n):
            for ng, cnt in hyp_ngrams[i].items():
                max_ref = max(r[ng] for r in ref_ngrams[i]) if ref_ngrams[i] else 0
                clip_counts[i] += min(cnt, max_ref)
                total_counts[i] += cnt
    # modified precisions
    precisions = [(clip_counts[i] / total_counts[i]) if total_counts[i] > 0 else 0.0 for i in range(max_n)]
    # brevity penalty
    if hyp_len == 0: return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    # geometric mean
    if any(p == 0 for p in precisions):
        geo = 0.0
    else:
        s = sum([weights[i] * math.log(precisions[i]) for i in range(max_n)])
        geo = math.exp(s)
    return bp * geo

def cer(ref: str, hyp: str) -> float:
    # character error rate = edit_distance / len(ref)
    def edit_distance(a: str, b: str) -> int:
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1): dp[i][0] = i
        for j in range(len(b)+1): dp[0][j] = j
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
        return dp[-1][-1]
    if len(ref) == 0: return 0.0 if len(hyp) == 0 else 1.0
    return edit_distance(ref, hyp) / len(ref)

def decode_sequence(tok: BPETokenizer, ids: List[int]) -> str:
    # Use BPETokenizer's decode for correctness
    return tok.decode([int(x) for x in ids])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--ckpt", type=str, default="checkpoints/bilstm_seq2seq.pt")
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--samples_out", type=str, default="results/samples.tsv")
    ap.add_argument("--metrics_out", type=str, default="results/metrics.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("results").mkdir(parents=True, exist_ok=True)

    # Data
    blob = torch.load(Path(args.data_dir) / f"{args.split}.pt", map_location="cpu")
    src = blob["src"]; tgt = blob["tgt"]; src_len = blob["src_len"]; tgt_len = blob["tgt_len"]

    # Tokenizers
    tok_src = BPETokenizer.load(Path(args.data_dir) / "bpe_src.json")
    tok_tgt = BPETokenizer.load(Path(args.data_dir) / "bpe_tgt.json")
    pad_tgt = tok_tgt.vocab[PAD]; bos_tgt = tok_tgt.vocab[BOS]; eos_tgt = tok_tgt.vocab[EOS]

    # Model (restore shapes from ckpt args)
    ckpt = torch.load(args.ckpt, map_location=device)
    a = ckpt["args"]
    model = Seq2Seq(
        src_vocab=ckpt["Vsrc"], tgt_vocab=ckpt["Vtgt"],
        src_pad=ckpt["pad_src"], tgt_pad=ckpt["pad_tgt"],
        emb_src=a["emb_src"], emb_tgt=a["emb_tgt"],
        enc_hidden=a["enc_hidden"], dec_hidden=a["dec_hidden"],
        enc_layers=a["enc_layers"], dec_layers=a["dec_layers"], dropout=a["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Perplexity of this split (token-level)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(src, tgt, src_len, tgt_len),
        batch_size=64, shuffle=False
    )
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for Xs, Ys, Ls, Lt in loader:
            Xs = Xs.to(device); Ys = Ys.to(device)
            Ls = Ls.to(device).long()
            tgt_inp = Ys[:, :-1]; tgt_gold = Ys[:, 1:]
            logits = model(Xs, Ls, tgt_inp)
            V = logits.size(-1)
            loss = criterion(logits.view(-1, V), tgt_gold.view(-1))
            mask = tgt_gold.ne(pad_tgt)
            total_loss += loss.item() * mask.sum().item()
            total_tok  += mask.sum().item()
    ppl = math.exp(total_loss / max(1, total_tok))

    # Decode + metrics (BLEU & CER)
    all_refs_tok, all_hyps_tok = [], []
    examples = []
    with torch.no_grad():
        for i in range(0, src.size(0), 64):
            Xs = src[i:i+64].to(device)
            Ls = src_len[i:i+64].to(device).long()
            pred_ids = greedy_decode(model, Xs, Ls, args.max_len, bos_tgt, eos_tgt, pad_tgt, device)  # (B,Tpred)
            Ygold = tgt[i:i+64][:, 1:]  # drop BOS
            # convert to python lists (strip PAD/EOS)
            for k in range(pred_ids.size(0)):
                hyp_seq = pred_ids[k].tolist()
                ref_seq = Ygold[k].tolist()
                # truncate at first EOS
                if eos_tgt in hyp_seq:
                    hyp_seq = hyp_seq[:hyp_seq.index(eos_tgt)]
                if eos_tgt in ref_seq:
                    ref_seq = ref_seq[:ref_seq.index(eos_tgt)]
                hyp_text = decode_sequence(tok_tgt, hyp_seq)
                ref_text = decode_sequence(tok_tgt, ref_seq)
                all_hyps_tok.append(list(hyp_text.split()))
                all_refs_tok.append(list(ref_text.split()))
                examples.append((i+k, hyp_text, ref_text))

    bleu = bleu_corpus(all_refs_tok, all_hyps_tok, max_n=4)
    # CER is char-level average over samples
    cer_vals = [cer(ref=" ".join(r), hyp=" ".join(h)) for r, h in zip(all_refs_tok, all_hyps_tok)]
    cer_avg = sum(cer_vals) / max(1, len(cer_vals))

    # save qualitative examples
    with open(args.samples_out, "w", encoding="utf-8") as f:
        f.write("idx\tpred\tgold\n")
        for idx, pred, gold in examples[:200]: 
            f.write(f"{idx}\t{pred}\t{gold}\n")

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "perplexity": ppl, "BLEU": bleu, "CER": cer_avg}, f, indent=2, ensure_ascii=False)

    print(f"{args.split.upper()} metrics ->  PPL: {ppl:.3f}  |  BLEU: {bleu:.4f}  |  CER: {cer_avg:.4f}")
    print(f"Saved examples -> {args.samples_out}")
    print(f"Saved metrics  -> {args.metrics_out}")

if __name__ == "__main__":
    main()
