"""Generate genuine Romanian speech with a community XTTS-v2 fine-tune (Path B).

XTTS-v2's base model does not support Romanian, but community fine-tunes do. This script
downloads such a fine-tune from the HuggingFace Hub, runs it fully on CPU, and clones a
voice from a short reference clip (or uses one of the fine-tune's built-in speakers).

CPU is slow: expect around real-time at best (no GPU acceleration here). This path is
independent of the pocket-tts model. The heavy lifting lives in
`pocket_tts.romanian.engines.XttsRomanianEngine`.

Requirements:
  pip install -e ".[romanian-xtts]"

Usage:
  python scripts/generate_romanian_xtts.py --speaker-wav path/to/romanian_reference.wav
  python scripts/generate_romanian_xtts.py --voice <builtin_speaker_id>
  python scripts/generate_romanian_xtts.py --text "Bună ziua." --output-path out.wav

Notes:
- Model weights are XTTS-v2 fine-tunes under the Coqui Public Model License (CPML).
- The default model repo is a community fine-tune; verify its terms before production use.
"""

import argparse
import logging
from pathlib import Path

import scipy.io.wavfile

from pocket_tts.romanian.engines import DEFAULT_XTTS_MODEL_REPO, XttsRomanianEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Romanian speech with a community XTTS-v2 fine-tune (CPU)."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Bună ziua, acesta este un test în limba română.",
        help="Romanian text to synthesize.",
    )
    parser.add_argument(
        "--speaker-wav",
        type=str,
        default=None,
        help="Path to a reference WAV (10-30s) of the voice to clone.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Built-in speaker id from the fine-tune's catalog (used if --speaker-wav is unset).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("tts_outputs/romanian_xtts/output.wav"),
        help="Where to write the generated WAV.",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=DEFAULT_XTTS_MODEL_REPO,
        help=f"HuggingFace repo of the XTTS-v2 Romanian fine-tune (default: {DEFAULT_XTTS_MODEL_REPO})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = XttsRomanianEngine(model_repo=args.model_repo)
    sample_rate, wav = engine.synthesize(
        args.text, voice=args.voice, speaker_wav=args.speaker_wav
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(args.output_path), sample_rate, wav)
    logger.info("Saved %s", args.output_path)


if __name__ == "__main__":
    main()
