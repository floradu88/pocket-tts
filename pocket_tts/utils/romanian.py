"""Romanian text helpers for the Italian-base pocket-tts path.

pocket-tts has no Romanian model. The `romanian` config reuses the Italian model and
its Italian SentencePiece tokenizer, which was never trained on Romanian-specific
letters (ă â î ș ț). Feeding those characters directly produces poor tokenization and
pronunciation, so we rewrite them into sequences the Italian tokenizer handles well.

This is a lossy, best-effort approximation - it yields an Italian-accented rendering of
Romanian, not correct native Romanian. For genuine Romanian use the XTTS-v2 path
(scripts/generate_romanian_xtts.py).
"""

# Legacy cedilla code points still appear in older Romanian text. Normalize them to the
# correct comma-below characters first so a single downstream mapping covers both.
_CEDILLA_TO_COMMA_BELOW = {
    "\u0163": "\u021b",  # ţ -> ț
    "\u0162": "\u021a",  # Ţ -> Ț
    "\u015f": "\u0219",  # ş -> ș
    "\u015e": "\u0218",  # Ş -> Ș
}

# Approximate Romanian letters with what the Italian tokenizer can pronounce.
# These are intentionally simple and predictable rather than phonetically perfect.
_ROMANIAN_TO_ITALIAN = {
    "ă": "a",
    "Ă": "A",
    "â": "a",
    "Â": "A",
    "î": "i",
    "Î": "I",
    "ș": "s",
    "Ș": "S",
    "ț": "ts",
    "Ț": "Ts",
}


def normalize_cedilla(text: str) -> str:
    """Map legacy cedilla diacritics (ţ, ş) to comma-below forms (ț, ș)."""
    return text.translate(str.maketrans(_CEDILLA_TO_COMMA_BELOW))


def normalize_for_italian_tokenizer(text: str) -> str:
    """Rewrite Romanian-specific letters into Italian-tokenizer-friendly sequences.

    The result is intelligible but Italian-accented Romanian. Multi-character
    replacements (e.g. ț -> ts) mean this cannot be done with str.translate alone.
    """
    text = normalize_cedilla(text)
    for src, dst in _ROMANIAN_TO_ITALIAN.items():
        text = text.replace(src, dst)
    return text
