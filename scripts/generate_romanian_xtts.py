"""Generate genuine Romanian speech with a community XTTS-v2 fine-tune (Path B).

XTTS-v2's base model does not support Romanian, but community fine-tunes do. This script
downloads such a fine-tune from the HuggingFace Hub, runs it fully on CPU, and clones a
voice from a short reference clip.

CPU is slow: expect around real-time at best (no GPU acceleration here). This path is
independent of the pocket-tts model.

Requirements:
  uv pip install -e ".[romanian-xtts]"

Usage:
  uv run python scripts/generate_romanian_xtts.py \
      --text "Bună ziua, acesta este un test în limba română." \
      --speaker-wav path/to/romanian_reference.wav \
      --output-path tts_outputs/romanian_xtts/output.wav

Notes:
- Model weights are XTTS-v2 fine-tunes under the Coqui Public Model License (CPML).
- The default model repo is a community fine-tune; verify its terms before production use.
- Romanian was not in XTTS-v2's original language set, so there is no explicit stop
  token. We enable text splitting to reduce hallucination/truncation on long inputs.
"""

import argparse
import logging
from pathlib import Path

from pocket_tts.utils.romanian import normalize_cedilla

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Community Romanian fine-tunes of XTTS-v2.
#   eduardem/xtts-v2-romanian-v2 : ~12.4% WER, 15 voices
#   eduardm/romanian-tts-xtts-v2 : ~6.6% WER (weights may live on Codeberg)
DEFAULT_MODEL_REPO = "eduardem/xtts-v2-romanian-v2"
LANGUAGE_CODE = "ro"


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
        required=True,
        help="Path to a reference WAV (10-30s) of the voice to clone.",
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
        default=DEFAULT_MODEL_REPO,
        help=f"HuggingFace repo of the XTTS-v2 Romanian fine-tune (default: {DEFAULT_MODEL_REPO})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional cap on generated GPT tokens to mitigate the missing Romanian stop token.",
    )
    return parser.parse_args()


def _ensure_audio_loader() -> None:
    """Make torchaudio.load work without FFmpeg by falling back to soundfile.

    torchaudio >= 2.9 routes audio IO through torchcodec, which needs FFmpeg shared
    libraries. Those are often missing on Windows. For WAV/FLAC/OGG reference clips we can
    decode with soundfile instead, so we monkeypatch torchaudio.load when torchcodec fails
    to load.
    """
    import torch
    import torchaudio

    try:
        import torchcodec.decoders  # noqa: F401

        return  # torchcodec works; nothing to do.
    except Exception:
        pass

    try:
        import soundfile as sf
    except ImportError:
        logger.warning(
            "torchcodec/FFmpeg unavailable and soundfile is not installed; "
            "reference audio decoding may fail. Install FFmpeg or soundfile."
        )
        return

    def _soundfile_load(filepath, *args, **kwargs):
        data, sample_rate = sf.read(str(filepath), dtype="float32", always_2d=True)
        # soundfile returns (samples, channels); torchaudio expects (channels, samples).
        return torch.from_numpy(data.T).contiguous(), sample_rate

    torchaudio.load = _soundfile_load
    logger.info("Using soundfile fallback for audio loading (FFmpeg/torchcodec unavailable).")


def _load_model(model_dir: Path):
    """Load an XTTS-v2 checkpoint on CPU. Imports Coqui TTS lazily with a clear error."""
    try:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except ImportError as exc:
        raise SystemExit(
            "Coqui TTS is not installed. Install the optional dependency group with:\n"
            '  uv pip install -e ".[romanian-xtts]"'
        ) from exc

    _ensure_audio_loader()

    config = XttsConfig()
    config.load_json(str(model_dir / "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), use_deepspeed=False)
    # Force CPU: never call model.cuda().
    model.to(torch.device("cpu"))
    model.eval()

    _patch_tokenizer_for_romanian(model)
    return model, config


def _patch_tokenizer_for_romanian(model) -> None:
    """Teach the base XTTS tokenizer about Romanian.

    The stock coqui-tts tokenizer predates Romanian, so:
    - `char_limits` has no 'ro' entry (text splitting -> KeyError); mirror Italian's limit.
    - `preprocess_text` rejects 'ro' outright, and the full multilingual cleaner relies on
      several per-language dicts (abbreviations, symbols, ordinal regexes) that lack 'ro'.
      We install a minimal cleaner for 'ro' (strip quotes, lowercase, collapse whitespace).
      Note: this skips number/abbreviation expansion for Romanian text.
    """
    from TTS.tts.layers.xtts import tokenizer as xtts_tok

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return

    char_limits = getattr(tokenizer, "char_limits", None)
    if isinstance(char_limits, dict) and LANGUAGE_CODE not in char_limits:
        char_limits[LANGUAGE_CODE] = char_limits.get("it", 250)
        logger.info("Added char limit for '%s': %d", LANGUAGE_CODE, char_limits[LANGUAGE_CODE])

    original_preprocess = tokenizer.preprocess_text

    def preprocess_text(txt, lang):
        if lang.split("-")[0] == LANGUAGE_CODE:
            txt = txt.replace('"', "")
            txt = xtts_tok.lowercase(txt)
            txt = xtts_tok.collapse_whitespace(txt)
            return txt
        return original_preprocess(txt, lang)

    tokenizer.preprocess_text = preprocess_text
    logger.info("Installed minimal Romanian text cleaner in the XTTS tokenizer.")


def main() -> None:
    args = parse_args()

    from huggingface_hub import snapshot_download

    logger.info("Downloading model %s ...", args.model_repo)
    model_dir = Path(
        snapshot_download(
            repo_id=args.model_repo,
            allow_patterns=["*.json", "*.pth", "*.model", "vocab.json", "speakers_xtts.pth"],
        )
    )

    logger.info("Loading XTTS-v2 model on CPU (this can take a while)...")
    model, config = _load_model(model_dir)

    # XTTS Romanian fine-tunes expect comma-below diacritics (ș, ț), not legacy cedillas.
    text = normalize_cedilla(args.text)

    synth_kwargs = dict(enable_text_splitting=True)
    if args.max_new_tokens is not None:
        synth_kwargs["gpt_max_new_tokens"] = args.max_new_tokens

    logger.info("Synthesizing (CPU, expect ~real-time or slower)...")
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=args.speaker_wav,
        language=LANGUAGE_CODE,
        **synth_kwargs,
    )

    import numpy as np
    import scipy.io.wavfile

    wav = np.asarray(outputs["wav"], dtype=np.float32)
    sample_rate = getattr(config, "output_sample_rate", 24000)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(args.output_path), sample_rate, wav)
    logger.info("Saved %s", args.output_path)


if __name__ == "__main__":
    main()
