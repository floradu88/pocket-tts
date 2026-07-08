"""Romanian TTS engines and API service.

Two CPU engines are provided:
- XttsRomanianEngine (primary): a community XTTS-v2 Romanian fine-tune, native pronunciation.
- PocketRomanianEngine (secondary): the Italian pocket-tts model with diacritic normalization.
"""

from pocket_tts.romanian.engines import (
    LANGUAGE_CODE,
    PocketRomanianEngine,
    RomanianEngine,
    XttsRomanianEngine,
)

__all__ = [
    "LANGUAGE_CODE",
    "RomanianEngine",
    "XttsRomanianEngine",
    "PocketRomanianEngine",
]
