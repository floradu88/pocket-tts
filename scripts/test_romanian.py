"""Smoke-test both Romanian generation paths and write output WAVs.

Runs the two scripts exactly as a user would (via subprocess, so Romanian diacritics in
argv are passed correctly on Windows):

  Path A -> scripts/generate_romanian.py       -> tts_outputs/romanian/test_pathA.wav
  Path B -> scripts/generate_romanian_xtts.py  -> tts_outputs/romanian_xtts/test_pathB.wav

Path A always runs (downloads the Italian model on first use). Path B needs the optional
`romanian-xtts` extra and a reference voice clip; if Coqui TTS is not installed it is
skipped with instructions instead of failing.

Usage:
  python scripts/test_romanian.py
  python scripts/test_romanian.py --skip-xtts
  python scripts/test_romanian.py --speaker-wav path/to/ref.wav
"""

import argparse
import logging
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_TEXT = (
    "Bună ziua! Acesta este un test în limba română, "
    "cu diacritice: șură, țară, când, întâlnire. Sper să sune bine."
)
PATH_A_OUT = REPO_ROOT / "tts_outputs" / "romanian" / "test_pathA.wav"
PATH_B_OUT = REPO_ROOT / "tts_outputs" / "romanian_xtts" / "test_pathB.wav"


def _utf8_env() -> dict:
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")


def _find_default_speaker_wav() -> Path | None:
    """Pick an existing WAV to use as an XTTS reference clip if the user gave none."""
    preferred = REPO_ROOT / "tts_outputs" / "italian" / "giovanni.wav"
    if preferred.exists():
        return preferred
    for candidate in sorted((REPO_ROOT / "tts_outputs").rglob("*.wav")):
        if candidate.name.startswith("test_path"):
            continue
        return candidate
    return None


def run_path_a() -> bool:
    logger.info("=== Path A: pocket-tts Italian base model ===")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_romanian.py"),
        "--text",
        TEST_TEXT,
        "--output-path",
        str(PATH_A_OUT),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_utf8_env())
    ok = result.returncode == 0 and PATH_A_OUT.exists()
    logger.info("Path A %s -> %s", "OK" if ok else "FAILED", PATH_A_OUT)
    return ok


def run_path_b(speaker_wav: Path | None) -> bool:
    logger.info("=== Path B: community XTTS-v2 Romanian fine-tune (CPU) ===")
    if find_spec("TTS") is None:
        logger.warning(
            "Coqui TTS not installed; skipping Path B. To enable it:\n"
            '  pip install -e ".[romanian-xtts]"   (or: uv pip install -e ".[romanian-xtts]")'
        )
        return False
    if speaker_wav is None or not speaker_wav.exists():
        logger.warning(
            "No reference speaker WAV found; skipping Path B. "
            "Provide one with --speaker-wav path/to/ref.wav"
        )
        return False

    logger.info("Using reference speaker clip: %s", speaker_wav)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_romanian_xtts.py"),
        "--text",
        TEST_TEXT,
        "--speaker-wav",
        str(speaker_wav),
        "--output-path",
        str(PATH_B_OUT),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_utf8_env())
    ok = result.returncode == 0 and PATH_B_OUT.exists()
    logger.info("Path B %s -> %s", "OK" if ok else "FAILED/SKIPPED", PATH_B_OUT)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test both Romanian TTS paths.")
    parser.add_argument("--skip-xtts", action="store_true", help="Skip Path B (XTTS-v2).")
    parser.add_argument(
        "--speaker-wav",
        type=Path,
        default=None,
        help="Reference voice clip for Path B (defaults to an existing tts_outputs WAV).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    a_ok = run_path_a()

    if args.skip_xtts:
        logger.info("Path B skipped (--skip-xtts).")
        b_ok = None
    else:
        speaker = args.speaker_wav or _find_default_speaker_wav()
        b_ok = run_path_b(speaker)

    logger.info("---- Summary ----")
    logger.info("Path A: %s", "OK" if a_ok else "FAILED")
    logger.info("Path B: %s", "skipped" if b_ok is None else ("OK" if b_ok else "FAILED/SKIPPED"))

    # Only Path A is required to succeed for the smoke test to pass.
    raise SystemExit(0 if a_ok else 1)


if __name__ == "__main__":
    main()
