# -*- coding: utf-8 -*-
"""
Data augmentation for Urdu→Roman Urdu transliteration.
Outputs: data/parallel_ur_roman_aug.tsv (original + synthetic)
Usage:
  python3 bonus/augment_data.py \
    --in_tsv data/parallel_ur_roman.tsv \
    --out_tsv data/parallel_ur_roman_aug.tsv \
    --roman_noise_prob 0.15 --roman_dup 1 \
    --backtrans_dup 1
"""
from __future__ import annotations
import argparse, random, re
from pathlib import Path

random.seed(42)

# --- Minimal normalization (subset of prep.py) ---
ARABIC_DIAC = "".join([
    "\u0610","\u0611","\u0612","\u0613","\u0614","\u0615","\u0616","\u0617","\u0618","\u0619","\u061A",
    "\u064B","\u064C","\u064D","\u064E","\u064F","\u0650","\u0651","\u0652","\u0653","\u0654","\u0655",
    "\u0656","\u0657","\u0658","\u0659","\u065A","\u065B","\u065C","\u065D","\u065E","\u065F","\u0670"
])
DIAC_RE = re.compile(f"[{ARABIC_DIAC}]")
TATWEEL_RE = re.compile(r"\u0640+")
MULTISPACE_RE = re.compile(r"\s+")
CANON = {"ي":"ی","ى":"ی","ئ":"ی","ۀ":"ہ","ة":"ہ","ۃ":"ہ","ك":"ک","أ":"ا","إ":"ا","آ":"ا","ؤ":"و"}
URDU_DIG = "۰۱۲۳۴۵۶۷۸۹"
ASCII_DIG = "0123456789"
D_MAP = {u:a for u,a in zip(URDU_DIG, ASCII_DIG)}
PUNCT = {"۔":".","،":",","؛":";","؟":"?","“":'"',"”":'"',"‘":"'", "’":"'"}

def normalize_urdu(s: str) -> str:
    s = DIAC_RE.sub("", s)
    s = TATWEEL_RE.sub("", s)
    s = "".join(CANON.get(ch, ch) for ch in s)
    s = "".join(D_MAP.get(ch, ch) for ch in s)
    s = "".join(PUNCT.get(ch, ch) for ch in s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

# --- Urdu→Roman (same spirit as prep.py fallback, shortened) ---
BIGRAMS = {"بھ":"bh","پھ":"ph","تھ":"th","ٹھ":"th","دھ":"dh","ڈھ":"dh","جھ":"jh","چھ":"chh","کھ":"kh","گھ":"gh","ڑھ":"rh"}
CMAP = {"ا":"a","ب":"b","پ":"p","ت":"t","ٹ":"t","ث":"s","ج":"j","چ":"ch","ح":"h","خ":"kh",
        "د":"d","ڈ":"d","ذ":"z","ر":"r","ڑ":"r","ز":"z","ژ":"zh","س":"s","ش":"sh","ص":"s","ض":"z",
        "ط":"t","ظ":"z","ع":"a","غ":"gh","ف":"f","ق":"q","ک":"k","گ":"g","ل":"l","م":"m","ن":"n","ں":"n",
        "و":"o","ہ":"h","ھ":"h","ء":"","ی":"i","ے":"e"}

def urdu_to_roman(s: str) -> str:
    s = normalize_urdu(s)
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace() or ch in ",.;:/?!'\"()-[]{}<>":
            out.append(ch); i += 1; continue
        if i+1 < len(s) and s[i:i+2] in BIGRAMS:
            out.append(BIGRAMS[s[i:i+2]]); i += 2; continue
        out.append(CMAP.get(ch, ch)); i += 1
    roman = "".join(out)
    roman = re.sub(r"aa+","a",roman); roman = re.sub(r"ii+","i",roman); roman = re.sub(r"oo+","o",roman)
    roman = re.sub(r"([bcdfghjklmnpqrstvxz])o([bcdfghjklmnpqrstvxz])", r"\1w\2", roman)
    roman = re.sub(r"([bcdfghjklmnpqrstvxz])i([bcdfghjklmnpqrstvxz])", r"\1y\2", roman)
    return MULTISPACE_RE.sub(" ", roman).strip()

# --- Roman→Urdu (coarse back-transliterator for augmentation only) ---
# Ordered multi-char to single-char mappings
R2U_MULTI = [
    ("shh","شہ"), ("chh","چھ"), ("kh","خ"), ("gh","غ"), ("zh","ژ"),
    ("bh","بھ"), ("ph","پھ"), ("th","تھ"), ("dh","دھ"), ("jh","جھ"), ("rh","ڑھ"), ("ch","چ"), ("sh","ش")
]
R2U_CHAR = {
    "a":"ا","b":"ب","p":"پ","t":"ت","d":"د","j":"ج","h":"ہ","k":"ک","g":"گ","z":"ز","s":"س","f":"ف","q":"ق",
    "l":"ل","m":"م","n":"ن","r":"ر","w":"و","o":"و","u":"و","i":"ی","e":"ے","y":"ی","x":"کس","c":"ک"
}
def roman_to_urdu(s: str) -> str:
    s = s.strip().lower()
    # multi-char first
    for pat, rep in R2U_MULTI:
        s = s.replace(pat, rep)
    # single-char
    out = []
    for ch in s:
        if ch.isspace():
            out.append(" ")
        elif ch in ",.;:/?!'\"()-[]{}<>":
            out.append(ch)
        else:
            out.append(R2U_CHAR.get(ch, ch))
    ur = "".join(out)
    ur = MULTISPACE_RE.sub(" ", ur)
    return normalize_urdu(ur)

# --- Noisers on Roman target ---
VOWELS = "aeiou"
def noise_roman(s: str, p: float=0.15) -> str:
    # with prob p per char, apply one of: swap, drop, vowel jitter, space jitter
    chars = list(s)
    i = 0
    while i < len(chars):
        if random.random() < p:
            op = random.choice(["swap","drop","vowel","space"])
            if op == "swap" and i+1 < len(chars) and chars[i].isalnum() and chars[i+1].isalnum():
                chars[i], chars[i+1] = chars[i+1], chars[i]; i += 2; continue
            if op == "drop" and chars[i].isalnum():
                del chars[i]; continue
            if op == "vowel" and chars[i].lower() in VOWELS:
                chars[i] = random.choice(VOWELS)
            if op == "space":
                if chars[i] == " " and random.random() < 0.5:
                    del chars[i]; continue
                elif chars[i].isalnum() and random.random() < 0.5:
                    chars.insert(i+1, " "); i += 2; continue
        i += 1
    out = "".join(chars)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def read_pairs(p: Path):
    pairs = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                ur, ro = line.rstrip("\n").split("\t")
                pairs.append((ur, ro))
            except ValueError:
                pass
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", type=str, default="data/parallel_ur_roman.tsv")
    ap.add_argument("--out_tsv", type=str, default="data/parallel_ur_roman_aug.tsv")
    ap.add_argument("--roman_noise_prob", type=float, default=0.15)
    ap.add_argument("--roman_dup", type=int, default=1, help="# noisy roman variants per original pair")
    ap.add_argument("--backtrans_dup", type=int, default=1, help="# back-trans pairs per original roman")
    args = ap.parse_args()

    pairs = read_pairs(Path(args.in_tsv))
    out = []

    for ur, ro in pairs:
        ur_n = normalize_urdu(ur)
        ro = ro.strip()
        if not ur_n or not ro: continue
        # original
        out.append((ur_n, ro))
        # roman noise
        for _ in range(max(0, args.roman_dup)):
            out.append((ur_n, noise_roman(ro, args.roman_noise_prob)))
        # back-trans: roman -> (synthetic urdu), pair with original roman
        for _ in range(max(0, args.backtrans_dup)):
            ur_bt = roman_to_urdu(ro)
            if ur_bt:
                out.append((ur_bt, ro))

    # write
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tsv, "w", encoding="utf-8") as f:
        for u, r in out:
            f.write(u + "\t" + r + "\n")
    print(f"Wrote {len(out)} pairs -> {args.out_tsv}")

if __name__ == "__main__":
    main()
