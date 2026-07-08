"""Generate WAV files for every built-in voice across every language config.

Uses each language's default phrase from default_parameters.py.

Usage:
  uv run python scripts/generate_all_languages_voices.py
  uv run python scripts/generate_all_languages_voices.py --languages english italian
  uv run python scripts/generate_all_languages_voices.py --voices alba giovanni --quantize
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

CONFIGS_DIR = Path("pocket_tts/config")
ALL_LANGUAGES = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
ALL_VOICES = list(_ORIGINS_OF_PREDEFINED_VOICES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate TTS for all languages and voices.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tts_outputs"),
        help="Directory for generated WAV files (default: tts_outputs)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=ALL_LANGUAGES,
        default=ALL_LANGUAGES,
        help="Language configs to generate (default: all)",
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
        help="Apply int8 quantization when loading models",
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

    jobs = [(language, voice) for language in args.languages for voice in args.voices]
    total = len(jobs)
    generated = 0
    skipped = 0
    failed: list[tuple[str, str, str]] = []

    current_language: str | None = None
    model = None

    for index, (language, voice) in enumerate(jobs, start=1):
        lang_dir = args.output_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        out_path = lang_dir / f"{voice}.wav"

        if args.skip_existing and out_path.exists():
            skipped += 1
            logger.info("[%d/%d] skip (exists): %s", index, total, out_path)
            continue

        if language != current_language:
            logger.info("Loading model for language: %s", language)
            model = TTSModel.load_model(language=language, quantize=args.quantize)
            current_language = language
            text = get_default_text_for_language(language)
            logger.info("Text: %s", text[:80] + ("..." if len(text) > 80 else ""))

        try:
            logger.info("[%d/%d] generating %s / %s -> %s", index, total, language, voice, out_path)
            voice_state = model.get_state_for_audio_prompt(voice)
            audio = model.generate_audio(voice_state, text)
            scipy.io.wavfile.write(str(out_path), model.sample_rate, audio.numpy())
            generated += 1
            logger.info("saved %s", out_path)
        except Exception as e:
            logger.error("Failed %s / %s: %s", language, voice, e)
            failed.append((language, voice, str(e)))

    logger.info(
        "Done. generated=%d skipped=%d failed=%d (total jobs=%d)",
        generated,
        skipped,
        len(failed),
        total,
    )
    if failed:
        for language, voice, err in failed:
            logger.error("  %s / %s: %s", language, voice, err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
