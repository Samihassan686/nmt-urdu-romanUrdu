# -*- coding: utf-8 -*-
from __future__ import annotations
import subprocess, json, itertools, os
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

GRID = [
    ("E1_base_256_512_2x4_do0.3_bs64_lr1e-3", {
        "emb_src":256, "emb_tgt":256, "enc_hidden":512, "dec_hidden":512,
        "enc_layers":2, "dec_layers":4, "dropout":0.3, "batch_size":64, "lr":1e-3, "epochs":20
    }),
    ("E2_small_128_256_1x3_do0.3_bs64_lr5e-4", {
        "emb_src":128, "emb_tgt":128, "enc_hidden":256, "dec_hidden":256,
        "enc_layers":1, "dec_layers":3, "dropout":0.3, "batch_size":64, "lr":5e-4, "epochs":20
    }),
    ("E3_big_512_512_3x4_do0.5_bs32_lr1e-4", {
        "emb_src":512, "emb_tgt":512, "enc_hidden":512, "dec_hidden":512,
        "enc_layers":3, "dec_layers":4, "dropout":0.5, "batch_size":32, "lr":1e-4, "epochs":20
    }),
]

def run_cmd(cmd: list[str]):
    print(">>", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True)

def main():
    csv = RESULTS_DIR / "experiments.csv"
    if not csv.exists():
        with open(csv, "w", encoding="utf-8") as f:
            f.write("timestamp,exp,split,ppl,bleu,cer,ckpt,samples,metrics\n")

    for exp_name, args in GRID:
        ckpt_path = CKPT_DIR / f"{exp_name}.pt"
        # 1) Train
        train_cmd = ["python3", "train_seq2seq.py",
            "--epochs", str(args["epochs"]),
            "--batch_size", str(args["batch_size"]),
            "--lr", str(args["lr"]),
            "--dropout", str(args["dropout"]),
            "--emb_src", str(args["emb_src"]),
            "--emb_tgt", str(args["emb_tgt"]),
            "--enc_hidden", str(args["enc_hidden"]),
            "--dec_hidden", str(args["dec_hidden"]),
            "--enc_layers", str(args["enc_layers"]),
            "--dec_layers", str(args["dec_layers"]),
        ]
        run_cmd(train_cmd)
        if (CKPT_DIR / "bilstm_seq2seq.pt").exists():
            os.replace(CKPT_DIR / "bilstm_seq2seq.pt", ckpt_path)

        # 2) Evaluate on val & test
        for split in ["val", "test"]:
            samples = RESULTS_DIR / f"{exp_name}_{split}_samples.tsv"
            metrics = RESULTS_DIR / f"{exp_name}_{split}_metrics.json"
            eval_cmd = ["python3", "infer_and_eval.py",
                "--split", split,
                "--ckpt", str(ckpt_path),
                "--samples_out", str(samples),
                "--metrics_out", str(metrics),
            ]
            run_cmd(eval_cmd)
            with open(metrics, "r", encoding="utf-8") as f:
                m = json.load(f)
            ts = datetime.now().isoformat(timespec="seconds")
            with open(csv, "a", encoding="utf-8") as f:
                f.write(f"{ts},{exp_name},{split},{m['perplexity']:.6f},{m['BLEU']:.6f},{m['CER']:.6f},{ckpt_path},{samples},{metrics}\n")

    print(f"All done. Summary -> {csv}")

if __name__ == "__main__":
    main()
