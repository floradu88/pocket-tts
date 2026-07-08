"""Romanian TTS engines.

- `XttsRomanianEngine` (primary): community XTTS-v2 Romanian fine-tune. Native Romanian
  pronunciation with voice cloning and a built-in speaker catalog. Heavy; CPU is slow.
- `PocketRomanianEngine` (secondary): the Italian pocket-tts model with Romanian diacritic
  normalization. Fast on CPU but Italian-accented.

Both expose a common `synthesize(text, voice=None, speaker_wav=None) -> (sample_rate, wav)`
interface where `wav` is a float32 numpy array.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pocket_tts.utils.romanian import normalize_cedilla, normalize_for_italian_tokenizer

logger = logging.getLogger(__name__)

LANGUAGE_CODE = "ro"
DEFAULT_XTTS_MODEL_REPO = "eduardem/xtts-v2-romanian-v2"
DEFAULT_POCKET_BASE_LANGUAGE = "italian"


class RomanianEngine:
    """Common interface for Romanian TTS engines."""

    name: str = "base"

    @property
    def is_loaded(self) -> bool:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def synthesize(
        self, text: str, voice: str | None = None, speaker_wav: str | None = None
    ) -> tuple[int, np.ndarray]:
        raise NotImplementedError


def _ensure_audio_loader() -> None:
    """Make torchaudio.load work without FFmpeg by falling back to soundfile.

    torchaudio >= 2.9 routes audio IO through torchcodec, which needs FFmpeg shared
    libraries (often absent on Windows). For WAV/FLAC/OGG clips we can decode with
    soundfile instead, so we monkeypatch torchaudio.load when torchcodec fails to load.
    """
    import torch
    import torchaudio

    try:
        import torchcodec.decoders  # noqa: F401

        return
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
        return torch.from_numpy(data.T).contiguous(), sample_rate

    torchaudio.load = _soundfile_load
    logger.info("Using soundfile fallback for audio loading (FFmpeg/torchcodec unavailable).")


class XttsRomanianEngine(RomanianEngine):
    """Primary engine: community XTTS-v2 Romanian fine-tune (CPU)."""

    name = "xtts"

    def __init__(
        self,
        model_repo: str = DEFAULT_XTTS_MODEL_REPO,
        default_speaker: str | None = None,
        default_speaker_wav: str | None = None,
        enable_text_splitting: bool = True,
    ) -> None:
        self.model_repo = model_repo
        self.default_speaker = default_speaker
        self.default_speaker_wav = default_speaker_wav
        self.enable_text_splitting = enable_text_splitting
        self._model = None
        self._config = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as exc:
            raise RuntimeError(
                "Coqui TTS is not installed. Install the optional dependency group with:\n"
                '  pip install -e ".[romanian-xtts]"'
            ) from exc

        from huggingface_hub import snapshot_download

        logger.info("Downloading XTTS Romanian model %s ...", self.model_repo)
        model_dir = Path(
            snapshot_download(
                repo_id=self.model_repo,
                allow_patterns=["*.json", "*.pth", "*.model", "vocab.json", "speakers_xtts.pth"],
            )
        )

        _ensure_audio_loader()

        logger.info("Loading XTTS-v2 model on CPU (this can take a while)...")
        config = XttsConfig()
        config.load_json(str(model_dir / "config.json"))

        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(model_dir), use_deepspeed=False)
        model.to(torch.device("cpu"))
        model.eval()

        self._patch_tokenizer_for_romanian(model)
        self._model = model
        self._config = config
        logger.info("XTTS Romanian model ready. Speakers: %s", self.available_speakers())

    @staticmethod
    def _patch_tokenizer_for_romanian(model) -> None:
        """Teach the base XTTS tokenizer about Romanian (char limit + minimal cleaner)."""
        from TTS.tts.layers.xtts import tokenizer as xtts_tok

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return

        char_limits = getattr(tokenizer, "char_limits", None)
        if isinstance(char_limits, dict) and LANGUAGE_CODE not in char_limits:
            char_limits[LANGUAGE_CODE] = char_limits.get("it", 250)

        original_preprocess = tokenizer.preprocess_text

        def preprocess_text(txt, lang):
            if lang.split("-")[0] == LANGUAGE_CODE:
                txt = txt.replace('"', "")
                txt = xtts_tok.lowercase(txt)
                txt = xtts_tok.collapse_whitespace(txt)
                return txt
            return original_preprocess(txt, lang)

        tokenizer.preprocess_text = preprocess_text

    def available_speakers(self) -> list[str]:
        speaker_manager = getattr(self._model, "speaker_manager", None)
        speakers = getattr(speaker_manager, "speakers", None)
        if isinstance(speakers, dict):
            return sorted(speakers.keys())
        return []

    def _resolve_voice(
        self, voice: str | None, speaker_wav: str | None
    ) -> tuple[str | None, str | None]:
        """Return (speaker_id, speaker_wav) to use, applying defaults/fallbacks."""
        if speaker_wav:
            return None, speaker_wav

        speakers = self.available_speakers()
        if voice and voice in speakers:
            return voice, None
        if voice:
            logger.warning("Unknown speaker '%s'; falling back to default.", voice)

        if self.default_speaker and self.default_speaker in speakers:
            return self.default_speaker, None
        if self.default_speaker_wav and Path(self.default_speaker_wav).exists():
            return None, self.default_speaker_wav
        if speakers:
            return speakers[0], None
        raise ValueError(
            "No speaker available: provide a speaker_wav clip or a valid speaker id."
        )

    def synthesize(
        self, text: str, voice: str | None = None, speaker_wav: str | None = None
    ) -> tuple[int, np.ndarray]:
        self.load()
        text = normalize_cedilla(text)
        speaker_id, resolved_wav = self._resolve_voice(voice, speaker_wav)

        logger.info(
            "XTTS synthesizing (speaker=%s, clip=%s)...",
            speaker_id,
            resolved_wav,
        )
        outputs = self._model.synthesize(
            text,
            speaker=speaker_id,
            speaker_wav=resolved_wav,
            language=LANGUAGE_CODE,
            enable_text_splitting=self.enable_text_splitting,
        )
        wav = np.asarray(outputs["wav"], dtype=np.float32)
        sample_rate = getattr(self._config, "output_sample_rate", 24000)
        return sample_rate, wav


class PocketRomanianEngine(RomanianEngine):
    """Secondary engine: Italian pocket-tts model + Romanian diacritic normalization."""

    name = "pocket"

    def __init__(
        self,
        base_language: str = DEFAULT_POCKET_BASE_LANGUAGE,
        default_voice: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.base_language = base_language
        self._default_voice = default_voice
        self.normalize = normalize
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from pocket_tts import TTSModel

        logger.info("Loading pocket-tts base model '%s' for Romanian...", self.base_language)
        self._model = TTSModel.load_model(language=self.base_language)

    def default_voice(self) -> str:
        if self._default_voice:
            return self._default_voice
        from pocket_tts.default_parameters import get_default_voice_for_language

        return get_default_voice_for_language(self.base_language)

    def synthesize(
        self, text: str, voice: str | None = None, speaker_wav: str | None = None
    ) -> tuple[int, np.ndarray]:
        self.load()
        if self.normalize:
            text = normalize_for_italian_tokenizer(text)

        chosen = speaker_wav or voice or self.default_voice()
        logger.info("pocket-tts synthesizing (voice=%s)...", chosen)
        voice_state = self._model.get_state_for_audio_prompt(chosen)
        audio = self._model.generate_audio(voice_state, text)
        wav = audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return int(self._model.sample_rate), wav
