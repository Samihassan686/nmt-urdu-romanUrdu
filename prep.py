# -*- coding: utf-8 -*-
"""
Preprocessing for Urdu → Roman Urdu parallel data (rekhta ghazals).
- Normalizes Urdu (diacritics, variants of Yeh/Heh, punctuation, tatweel, digits).
- Uses 'en/' peer files as Roman target when available; else applies a rule-based transliterator.
- Produces: data/parallel_ur_roman.tsv  (columns: ur<TAB>roman)
"""

import os
import re
import unicodedata
import random
from pathlib import Path

# -----------------------
# Urdu normalization
# -----------------------

ARABIC_DIACRITICS = "".join([
    "\u0610", "\u0611", "\u0612", "\u0613", "\u0614", "\u0615",
    "\u0616", "\u0617", "\u0618", "\u0619", "\u061A",
    "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652", "\u0653", "\u0654", "\u0655",
    "\u0656", "\u0657", "\u0658", "\u0659", "\u065A", "\u065B", "\u065C", "\u065D", "\u065E", "\u065F",
    "\u0670"
])

DIACRITICS_RE = re.compile(f"[{ARABIC_DIACRITICS}]")
TATWEEL_RE = re.compile(r"\u0640+")
MULTISPACE_RE = re.compile(r"\s+")
# keep Urdu punctuation that matters poetically but normalize spaces around them
PUNCT_MAP = {
    "۔": ".", "،": ",", "؛": ";", "؟": "?", "؍": "/", "٪": "%", "‹": "<", "›": ">",
    "“": '"', "”": '"', "‘": "'", "’": "'"
}

# Canonical character mapping (Urdu variants → preferred)
CANON_MAP = {
    # Yeh forms
    "ي": "ی", "ى": "ی", "ئ": "ی",  # hamza-over-yeh simplified
    # Heh forms
    "ۀ": "ہ", "ة": "ہ", "ۃ": "ہ",
    # Kaf/Farsi Kaf
    "ك": "ک",
    # Alef variants
    "أ": "ا", "إ": "ا", "آ": "ا",  # we simplify long alif to bare alef for modeling; length learned via subwords
    # Waw with hamza
    "ؤ": "و",
    # Teh marbuta already mapped above
}

URDU_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ASCII_DIGITS = "0123456789"
DIGIT_MAP = {u: a for u, a in zip(URDU_DIGITS, ASCII_DIGITS)}

def normalize_urdu(text: str) -> str:
    # NFC → strip diacritics → canonical char map → normalize punctuation/spaces
    text = unicodedata.normalize("NFC", text)
    text = DIACRITICS_RE.sub("", text)
    text = TATWEEL_RE.sub("", text)

    # map chars
    text = "".join(CANON_MAP.get(ch, ch) for ch in text)
    text = "".join(DIGIT_MAP.get(ch, ch) for ch in text)
    # punctuation normalization
    text = "".join(PUNCT_MAP.get(ch, ch) for ch in text)
    # collapse spaces and trim
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

# -----------------------
# Rule-based transliteration (fallback)
# -----------------------
# Very lightweight Urdu→Roman rules, good enough as a backstop.
# Note: The model will learn better spellings; this is just to bootstrap pairs when Roman lines are missing.

# Aspirated digraphs with do-chashmi heh
BIGRAMS = {
    "بھ": "bh", "پھ": "ph", "تھ": "th", "ٹھ": "th", "دھ": "dh", "ڈھ": "dh",
    "جھ": "jh", "چھ": "chh", "کھ": "kh", "گھ": "gh", "ڑھ": "rh", "شہ": "shh"  # last one rare
}

# Single letters (simplified)
CHAR_MAP = {
    "ا": "a", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch",
    "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r",
    "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z",
    "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "گ": "g", "ل": "l", "م": "m",
    "ن": "n", "ں": "n", "و": "o", "ہ": "h", "ھ": "h", "ء": "", "ی": "i", "ے": "e"
}

VOWEL_HEURISTICS = True  # add simple context fixes for و/ی

def urdu_to_roman_fallback(line: str) -> str:
    s = normalize_urdu(line)
    if not s:
        return s

    # handle digraphs first
    out = []
    i = 0
    while i < len(s):
        # skip spaces/punct as-is
        ch = s[i]
        if ch.isspace() or ch in ",.;:/?!'\"()-[]{}<>":
            out.append(ch)
            i += 1
            continue

        if i + 1 < len(s):
            pair = s[i : i + 2]
            if pair in BIGRAMS:
                out.append(BIGRAMS[pair])
                i += 2
                continue

        # map single char
        rom = CHAR_MAP.get(ch, ch)
        out.append(rom)
        i += 1

    roman = "".join(out)

    if VOWEL_HEURISTICS:
        # very light tweaks: collapse repeated vowels, basic w/y behavior
        roman = re.sub(r"aa+", "a", roman)
        roman = re.sub(r"ii+", "i", roman)
        roman = re.sub(r"oo+", "o", roman)
        # 'w' vs 'o/u' for و : if between consonants, prefer 'w'
        roman = re.sub(r"([bcdfghjklmnpqrstvxz])o([bcdfghjklmnpqrstvxz])", r"\1w\2", roman)
        # 'y' vs 'i/ee' for ی : if between consonants, prefer 'y'
        roman = re.sub(r"([bcdfghjklmnpqrstvxz])i([bcdfghjklmnpqrstvxz])", r"\1y\2", roman)

    # collapse multiple spaces again
    roman = MULTISPACE_RE.sub(" ", roman).strip()
    return roman

# -----------------------
# File crawling & pairing
# -----------------------

def read_lines(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        # strip blank/whitespace-only lines
        return [ln.strip("\n\r") for ln in f if ln.strip()]

def pair_for_poet(poet_dir: Path):
    """Yield (ur, roman) pairs for one poet folder with subdirs ur/, en/ (optional)."""
    ur_dir = poet_dir / "ur"
    en_dir = poet_dir / "en"  # may be roman transliteration or English
    have_en = en_dir.exists()

    # files by basename
    for ur_file in sorted(ur_dir.glob("*")):
        if not ur_file.is_file():
            continue
        base = ur_file.name
        ur_lines = read_lines(ur_file)
        roman_lines = None

        if have_en:
            en_file = en_dir / base
            if en_file.exists():
                roman_lines = read_lines(en_file)

        # Align by line index. If counts mismatch, take min length.
        if roman_lines is not None and len(roman_lines) > 0:
            n = min(len(ur_lines), len(roman_lines))
            for i in range(n):
                ur = normalize_urdu(ur_lines[i])
                ro = roman_lines[i].strip()
                # if 'en' looks like a translation (has many spaces/English words),
                # we still keep it; later filtering will decide. If it's clearly not roman,
                # fallback.
                if re.search(r"[A-Za-z]", ro):
                    yield ur, ro
                else:
                    yield ur, urdu_to_roman_fallback(ur)
        else:
            # No 'en' peer: fallback transliteration
            for ur in ur_lines:
                ur = normalize_urdu(ur)
                if ur:
                    yield ur, urdu_to_roman_fallback(ur)

def crawl_repo(repo_root: Path):
    """Traverse all poet folders under urdu_ghazals_rekhta/dataset/dataset/dataset/*"""
    target_root = repo_root / "urdu_ghazals_rekhta" / "dataset" / "dataset" / "dataset"
    assert target_root.exists(), f"Not found: {target_root}"
    all_pairs = []
    for poet in sorted(target_root.glob("*")):
        if not poet.is_dir():
            continue
        if not (poet / "ur").exists():
            continue
        for ur, ro in pair_for_poet(poet):
            if ur and ro:
                all_pairs.append((ur, ro))
    return all_pairs

# -----------------------
# Basic cleaning & filtering
# -----------------------

def looks_like_roman(s: str) -> bool:
    # Roman Urdu typically uses Latin letters, spaces, and punctuation
    return bool(re.search(r"[A-Za-z]", s)) and not bool(re.search(r"[\u0600-\u06FF]", s))

def filter_pairs(pairs):
    out = []
    seen = set()
    for ur, ro in pairs:
        ur = normalize_urdu(ur)
        ro = ro.strip()
        if not ur or not ro:
            continue
        # ensure roman doesn't contain Urdu script
        if not looks_like_roman(ro):
            # try fallback transliteration if 'en' was actually English translation
            ro = urdu_to_roman_fallback(ur)
            if not looks_like_roman(ro):
                continue
        key = (ur, ro.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((ur, ro))
    return out

# -----------------------
# Main
# -----------------------

def main():
    repo_root = Path(".").resolve()
    out_dir = repo_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Crawling dataset...")
    pairs = crawl_repo(repo_root)
    print(f"Raw pairs collected: {len(pairs)}")
    pairs = filter_pairs(pairs)
    print(f"After filtering/dedup: {len(pairs)}")

    # shuffle for randomness
    random.seed(42)
    random.shuffle(pairs)

    out_path = out_dir / "parallel_ur_roman.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        for ur, ro in pairs:
            f.write(ur + "\t" + ro + "\n")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
