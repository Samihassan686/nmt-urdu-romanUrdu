# -*- coding: utf-8 -*-
"""
A from-scratch BPE tokenizer with:
- train(): learn merges up to vocab_size from corpus (list of strings)
- encode()/decode()
- save()/load()
- uses a word-boundary marker '▁' like SentencePiece, but no external libs
- Unicode-safe for Urdu + Roman

This is intentionally small & clear; optimized for reproducible coursework.
"""
from __future__ import annotations
import json, re, collections
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

WB = "▁"  # word boundary marker
UNK, PAD, BOS, EOS = "<unk>", "<pad>", "<bos>", "<eos>"

def word_to_chars(word: str) -> Tuple[str, ...]:
    # Represent a token as tuple of "characters" (unicode codepoints)
    return tuple(list(word))

def whitespace_tokenize(text: str) -> List[str]:
    # keep punctuation as part of tokens; we already normalized in prep.py
    return text.strip().split()

def get_vocab_from_corpus(corpus: Iterable[str]) -> List[List[str]]:
    # Pre-tokenize text into "words" with boundary marker prefix
    # Each whitespace token becomes a word piece sequence starting with WB
    out = []
    for line in corpus:
        words = whitespace_tokenize(line)
        out.append([WB + w for w in words])
    return out

def compute_pair_stats(seqs: List[List[Tuple[str, ...]]]) -> Dict[Tuple[str, str], int]:
    stats = collections.Counter()
    for seq in seqs:
        for symbols in seq:
            for i in range(len(symbols) - 1):
                stats[(symbols[i], symbols[i+1])] += 1
    return stats

def merge_symbols(symbols: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    a, b = pair
    res = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
            res.append(a + b)
            i += 2
        else:
            res.append(symbols[i])
            i += 1
    return tuple(res)

class BPETokenizer:
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.id2tok: List[str] = []

    def _init_base_vocab(self, corpus_words: List[List[str]]) -> List[List[Tuple[str, ...]]]:
        # Represent each word as tuple of chars (including WB as first char)
        seqs = []
        for line_words in corpus_words:
            seq = []
            for w in line_words:
                seq.append(word_to_chars(w))
            seqs.append(seq)
        return seqs

    def train(self, corpus: Iterable[str], vocab_size: int, min_freq: int = 2, specials: List[str] = None):
        """
        Train BPE merges up to vocab_size.
        """
        specials = specials or [PAD, BOS, EOS, UNK]
        # base units are unicode chars; collect words
        corpus_words = get_vocab_from_corpus(corpus)
        seqs = self._init_base_vocab(corpus_words)

        # count initial symbol frequencies to form initial vocab
        sym_freq = collections.Counter()
        for seq in seqs:
            for symbols in seq:
                for s in symbols:
                    sym_freq[s] += 1

        merges = []
        # Keep merging most frequent adjacent pair until vocab_size reached
        # Target vocab size counts: base symbols + merges + specials
        def current_vocab_size():
            # approximate: number of unique symbols in sym_freq
            return len([s for s,f in sym_freq.items() if f >= 1]) + len(specials)

        while True:
            pair_stats = compute_pair_stats(seqs)
            if not pair_stats:
                break
            # filter by min_freq for stability
            pair, freq = pair_stats.most_common(1)[0]
            if freq < min_freq:
                break
            # stop if vocab size target would be exceeded
            if current_vocab_size() + 1 > vocab_size:
                break
            merges.append(pair)
            # apply merge to all sequences; update sym_freq roughly
            new_seqs = []
            for seq in seqs:
                new_seq = []
                for symbols in seq:
                    merged = merge_symbols(symbols, pair)
                    new_seq.append(merged)
                new_seqs.append(new_seq)
            seqs = new_seqs

            # recompute sym_freq cheaply
            sym_freq = collections.Counter()
            for seq in seqs:
                for symbols in seq:
                    for s in symbols:
                        sym_freq[s] += 1

        self.merges = merges

        # Build final vocab: specials + discovered symbols
        symbols = set()
        for seq in seqs:
            for symbols_tuple in seq:
                for s in symbols_tuple:
                    symbols.add(s)
        # Ensure WB itself is part of some symbols (it will be as prefix char)
        # Create deterministic order
        base_tokens = sorted(symbols)
        self.id2tok = specials + base_tokens
        self.vocab = {t: i for i, t in enumerate(self.id2tok)}

    def _apply_merges_to_word(self, word: str) -> List[str]:
        # Start from characters
        symbols = list(word)
        # Apply merges sequentially
        for a, b in self.merges:
            i = 0
            new_syms = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    new_syms.append(a + b)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms
        # Map symbols to IDs, falling back to UNK for unseen
        return symbols

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        tokens = []
        for w in whitespace_tokenize(text):
            word = WB + w  # add boundary
            pieces = self._apply_merges_to_word(word)
            for p in pieces:
                tokens.append(self.vocab.get(p, self.vocab.get(UNK)))
        if add_bos_eos:
            tokens = [self.vocab[BOS]] + tokens + [self.vocab[EOS]]
        return tokens

    def decode(self, ids: List[int]) -> str:
        toks = [self.id2tok[i] for i in ids if 0 <= i < len(self.id2tok)]
        # strip BOS/EOS if present
        toks = [t for t in toks if t not in (BOS, EOS, PAD)]
        # glue back words by merging pieces and removing WB
        text = ""
        buf = ""
        for t in toks:
            buf += t
        # Split back into words by WB
        words = [w.replace(WB, "") for w in re.split(f"({WB})", buf) if w and w != WB]
        return " ".join(words)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "merges": self.merges,
                "id2tok": self.id2tok,
                "WB": WB,
                "specials": [PAD, BOS, EOS, UNK]
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tok.merges = [tuple(x) for x in obj["merges"]]
        tok.id2tok = obj["id2tok"]
        tok.vocab = {t: i for i, t in enumerate(tok.id2tok)}
        return tok
