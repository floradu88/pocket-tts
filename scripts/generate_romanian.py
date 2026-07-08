"""Generate Romanian speech using a Romance pocket-tts model (Path A).

Romanian has no dedicated pocket-tts model. This script reuses the closest Romance model
(Italian by default) and normalizes Romanian-specific letters into sequences that model's
tokenizer can pronounce. The result is intelligible but foreign-accented Romanian.

Two voice modes:
- Predefined catalog voice (default, e.g. `giovanni`): works out of the box, no auth. The
  voice embedding is loaded from the base language's folder on HuggingFace.
- Voice cloning from a WAV (`--voice path/to/clip.wav`): gives a Romanian accent if you
  pass a Romanian clip, but requires access to the gated voice-cloning weights
  (accept terms at https://huggingface.co/kyutai/pocket-tts and `hf auth login`).

For genuine native pronunciation, use scripts/generate_romanian_xtts.py instead.

Usage:
  python scripts/generate_romanian.py
  python scripts/generate_romanian.py --text "Bună ziua, ce mai faceți?"
  python scripts/generate_romanian.py --voice path/to/romanian_clone.wav
  python scripts/generate_romanian.py --base-language spanish
  python scripts/generate_romanian.py --text - --output-path out.wav  # read stdin
"""

import argparse
import logging
import sys
from pathlib import Path

import scipy.io.wavfile

from pocket_tts import TTSModel
from pocket_tts.default_parameters import (
    get_default_text_for_language,
    get_default_voice_for_language,
)
from pocket_tts.utils.romanian import normalize_for_italian_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Romance languages that ship real weights + predefined voice embeddings. Romanian is
# closest to Italian, so that is the default base.
BASE_LANGUAGE_CHOICES = ["italian", "spanish", "portuguese", "french_24l"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Romanian speech with a Romance pocket-tts base model."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Romanian text to synthesize. Use '-' to read from stdin. "
        "Defaults to the built-in Romanian phrase.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice: a built-in name (e.g. giovanni), or a WAV path / hf:// url for cloning "
        "(cloning needs gated voice-cloning weights). Defaults to the base language's voice.",
    )
    parser.add_argument(
        "--base-language",
        type=str,
        default="italian",
        choices=BASE_LANGUAGE_CHOICES,
        help="Romance model to reuse for Romanian (default: italian).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("tts_outputs/romanian/output.wav"),
        help="Where to write the generated WAV (default: tts_outputs/romanian/output.wav)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not rewrite Romanian diacritics for the base tokenizer (not recommended).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply int8 quantization when loading the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    text = args.text
    if text is None:
        text = get_default_text_for_language("romanian")
    elif text == "-":
        text = sys.stdin.read()

    if not args.no_normalize:
        normalized = normalize_for_italian_tokenizer(text)
        if normalized != text:
            logger.info("Normalized text for the base tokenizer: %s", normalized)
        text = normalized

    # Reuse a real Romance language so predefined voice embeddings resolve without needing
    # the gated voice-cloning weights.
    voice = args.voice if args.voice is not None else get_default_voice_for_language(args.base_language)

    logger.info("Loading base model '%s' for Romanian...", args.base_language)
    model = TTSModel.load_model(language=args.base_language, quantize=args.quantize)

    logger.info("Building voice state from: %s", voice)
    voice_state = model.get_state_for_audio_prompt(voice)

    logger.info("Generating audio...")
    audio = model.generate_audio(voice_state, text)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(args.output_path), model.sample_rate, audio.numpy())
    logger.info("Saved %s", args.output_path)


if __name__ == "__main__":
    main()
