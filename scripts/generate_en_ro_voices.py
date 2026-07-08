"""Generate WAV files for English and Romanian across all built-in voices.

Both use the English Pocket TTS model:
- English: the built-in English default phrase.
- Romanian: a Romanian intro phrase read by the English model. Pocket TTS has no
  dedicated Romanian model, so pronunciation/accent will not be native.

Usage:
  python scripts/generate_en_ro_voices.py
  python scripts/generate_en_ro_voices.py --voices alba marius --skip-existing
"""

import argparse
import logging
from pathlib import Path

import scipy.io.wavfile

from pocket_tts import TTSModel
from pocket_tts.default_parameters import get_default_text_for_language
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_VOICES = list(_ORIGINS_OF_PREDEFINED_VOICES.keys())

# The English model is used for both languages (no dedicated Romanian model exists).
MODEL_LANGUAGE = "english"

ROMANIAN_TEXT = (
    "Salut lume. Sunt Pocket TTS de la Kyutai. "
    "Sunt suficient de rapid ca sa rulez pe procesoare mici. "
    "Sper sa ma placi."
)

# label -> text to speak
TARGETS = {
    "english": get_default_text_for_language("english"),
    "romanian": ROMANIAN_TEXT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate English + Romanian TTS for all voices.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tts_outputs"),
        help="Directory for generated WAV files (default: tts_outputs)",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        choices=ALL_VOICES,
        default=ALL_VOICES,
        help="Voices to generate (default: all)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply int8 quantization when loading the model",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations whose WAV already exists in the output folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(label, voice) for label in TARGETS for voice in args.voices]
    total = len(jobs)
    generated = 0
    skipped = 0
    failed: list[tuple[str, str, str]] = []

    model = None

    for index, (label, voice) in enumerate(jobs, start=1):
        lang_dir = args.output_dir / label
        lang_dir.mkdir(parents=True, exist_ok=True)
        out_path = lang_dir / f"{voice}.wav"

        if args.skip_existing and out_path.exists():
            skipped += 1
            logger.info("[%d/%d] skip (exists): %s", index, total, out_path)
            continue

        if model is None:
            logger.info("Loading %s model (used for both English and Romanian)", MODEL_LANGUAGE)
            model = TTSModel.load_model(language=MODEL_LANGUAGE, quantize=args.quantize)

        text = TARGETS[label]
        try:
            logger.info("[%d/%d] generating %s / %s -> %s", index, total, label, voice, out_path)
            voice_state = model.get_state_for_audio_prompt(voice)
            audio = model.generate_audio(voice_state, text)
            scipy.io.wavfile.write(str(out_path), model.sample_rate, audio.numpy())
            generated += 1
            logger.info("saved %s", out_path)
        except Exception as e:
            logger.error("Failed %s / %s: %s", label, voice, e)
            failed.append((label, voice, str(e)))

    logger.info(
        "Done. generated=%d skipped=%d failed=%d (total jobs=%d)",
        generated,
        skipped,
        len(failed),
        total,
    )
    if failed:
        for label, voice, err in failed:
            logger.error("  %s / %s: %s", label, voice, err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
