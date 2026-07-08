"""Shared runtime for the Romanian TTS API and MCP server.

Holds the engine singletons, configuration, a generation lock (models are not thread-safe
on CPU), and helpers for synthesizing to bytes or to files in the shared output directory.

Kept import-free of `service` and `mcp_server` to avoid circular imports.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from pocket_tts.romanian.engines import (
    DEFAULT_POCKET_BASE_LANGUAGE,
    DEFAULT_XTTS_MODEL_REPO,
    PocketRomanianEngine,
    XttsRomanianEngine,
)

logger = logging.getLogger(__name__)


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_ENGINE = os.environ.get("ROMANIAN_DEFAULT_ENGINE", "auto").lower()
MAX_CHARS = int(os.environ.get("ROMANIAN_MAX_CHARS", "1000"))
OUTPUT_DIR = Path(os.environ.get("ROMANIAN_OUTPUT_DIR", "tts_outputs/romanian_api"))
PUBLIC_BASE_URL = os.environ.get("ROMANIAN_PUBLIC_BASE_URL", "").rstrip("/")
CACHE_ENABLED = env_flag("ROMANIAN_CACHE", True)
CACHE_INDEX_PATH = OUTPUT_DIR / "cache_index.json"

xtts_engine = XttsRomanianEngine(
    model_repo=os.environ.get("ROMANIAN_MODEL_REPO", DEFAULT_XTTS_MODEL_REPO),
    default_speaker=os.environ.get("ROMANIAN_XTTS_SPEAKER") or None,
    default_speaker_wav=os.environ.get("ROMANIAN_DEFAULT_SPEAKER_WAV") or None,
)
pocket_engine = PocketRomanianEngine(
    base_language=os.environ.get("POCKET_BASE_LANGUAGE", DEFAULT_POCKET_BASE_LANGUAGE),
)
ENGINES = {"xtts": xtts_engine, "pocket": pocket_engine}
VALID_ENGINES = {"auto", "xtts", "pocket"}

_generation_lock = threading.Lock()


def preload(primary: bool | None = None, secondary: bool | None = None) -> None:
    """Preload engines according to args or ROMANIAN_PRELOAD_* env flags."""
    if secondary is None:
        secondary = env_flag("ROMANIAN_PRELOAD_SECONDARY", True)
    if primary is None:
        primary = env_flag("ROMANIAN_PRELOAD_PRIMARY", False)
    if secondary:
        try:
            pocket_engine.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to preload pocket engine: %s", exc)
    if primary:
        try:
            xtts_engine.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to preload XTTS engine: %s", exc)
    # Start the generation worker and load the persisted cache index up front so cache stats
    # are accurate immediately and the queue is warm before the first request.
    _ensure_worker()


def wav_bytes(sample_rate: int, wav: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, wav)
    return buffer.getvalue()


def duration_seconds(sample_rate: int, wav: np.ndarray) -> float:
    if sample_rate <= 0:
        return 0.0
    return round(len(np.asarray(wav).reshape(-1)) / sample_rate, 3)


def synthesize(engine_name: str, text: str, voice: str | None, speaker_wav: str | None):
    """Synthesize with a single named engine (serialized by the generation lock)."""
    engine = ENGINES[engine_name]
    with _generation_lock:
        return engine.synthesize(text, voice=voice, speaker_wav=speaker_wav)


def synthesize_with_fallback(
    requested: str, text: str, voice: str | None, speaker_wav: str | None
) -> tuple[str, int, np.ndarray]:
    """Synthesize honoring engine selection; `auto` prefers XTTS then falls back to pocket.

    Returns (engine_used, sample_rate, wav). Raises the last error if all engines fail.
    """
    order = ["xtts", "pocket"] if requested == "auto" else [requested]
    last_error: Exception | None = None
    for name in order:
        try:
            sample_rate, wav = synthesize(name, text, voice, speaker_wav)
            return name, sample_rate, wav
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Engine '%s' failed: %s", name, exc)
    raise RuntimeError(f"All requested engines failed: {last_error}")


def save_wav(sample_rate: int, wav: np.ndarray, filename: str | None = None) -> Path:
    """Write a WAV into the shared output directory and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = f"ro_{uuid.uuid4().hex[:12]}.wav"
    if not filename.lower().endswith(".wav"):
        filename += ".wav"
    filename = Path(filename).name  # prevent path traversal
    out_path = OUTPUT_DIR / filename
    scipy.io.wavfile.write(str(out_path), sample_rate, wav)
    return out_path


def public_url(filename: str) -> str | None:
    if not PUBLIC_BASE_URL:
        return None
    return f"{PUBLIC_BASE_URL}/outputs/{filename}"


# --------------------------------------------------------------------------------------
# Content cache + de-duplicating generation queue
#
# Goals:
# - Never synthesize the same (text, engine, voice, speaker clip) twice: results are cached
#   on disk and matched by a content hash, then served directly.
# - Never run two identical generations concurrently: in-flight requests for the same key
#   share a single Future.
# - Serialize all generation through a single worker thread (the models are not
#   thread-safe on CPU) — this is the "queue".
# --------------------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[str, dict] = {}

_jobs_lock = threading.Lock()
_inflight: dict[str, Future] = {}
_job_queue = queue.Queue()
_worker_started = threading.Event()


@dataclass
class _Job:
    key: str
    text: str
    engine: str
    voice: str | None
    speaker_wav: str | None
    filename: str | None
    future: Future = field(default_factory=Future)


def _fingerprint_speaker(speaker_wav: str | None) -> str:
    if not speaker_wav:
        return ""
    try:
        data = Path(speaker_wav).read_bytes()
    except OSError:
        return speaker_wav  # fall back to the path string
    return hashlib.sha256(data).hexdigest()[:16]


def cache_key(text: str, engine: str, voice: str | None, speaker_wav: str | None) -> str:
    payload = json.dumps(
        {
            "text": text,
            "engine": engine,
            "voice": voice or "",
            "speaker": _fingerprint_speaker(speaker_wav),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_cache_index() -> None:
    if not CACHE_ENABLED or not CACHE_INDEX_PATH.exists():
        return
    try:
        raw = json.loads(CACHE_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read cache index: %s", exc)
        return
    pruned = {
        key: entry
        for key, entry in raw.items()
        if (OUTPUT_DIR / entry.get("filename", "")).exists()
    }
    with _cache_lock:
        _cache.clear()
        _cache.update(pruned)
    logger.info("Loaded %d cached Romanian TTS entries.", len(pruned))


def _save_cache_index() -> None:
    if not CACHE_ENABLED:
        return
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_INDEX_PATH.write_text(
            json.dumps(_cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Could not write cache index: %s", exc)


def _cache_lookup(key: str) -> dict | None:
    if not CACHE_ENABLED:
        return None
    with _cache_lock:
        entry = _cache.get(key)
    if entry and (OUTPUT_DIR / entry["filename"]).exists():
        return entry
    return None


def _run_job(job: _Job) -> dict:
    # Re-check the cache in case an identical request completed while queued.
    entry = _cache_lookup(job.key)
    if entry is not None:
        return {**entry, "url": public_url(entry["filename"])}

    engine_used, sample_rate, wav = synthesize_with_fallback(
        job.engine, job.text, job.voice, job.speaker_wav
    )
    filename = job.filename or f"ro_{job.key}.wav"
    path = save_wav(sample_rate, wav, filename)
    entry = {
        "filename": path.name,
        "engine_used": engine_used,
        "sample_rate": sample_rate,
        "duration_s": duration_seconds(sample_rate, wav),
        "text": job.text[:500],
    }
    if CACHE_ENABLED:
        with _cache_lock:
            _cache[job.key] = entry
            _save_cache_index()
    return {**entry, "path": str(path.resolve()), "url": public_url(path.name)}


def _worker() -> None:
    while True:
        job = _job_queue.get()
        try:
            job.future.set_result(_run_job(job))
        except Exception as exc:  # noqa: BLE001
            job.future.set_exception(exc)
        finally:
            with _jobs_lock:
                _inflight.pop(job.key, None)
            _job_queue.task_done()


def _ensure_worker() -> None:
    if _worker_started.is_set():
        return
    with _jobs_lock:
        if _worker_started.is_set():
            return
        threading.Thread(target=_worker, name="romanian-tts-worker", daemon=True).start()
        _worker_started.set()
        _load_cache_index()


def queue_depth() -> int:
    """Approximate number of jobs waiting or running."""
    with _jobs_lock:
        return len(_inflight)


def cache_stats() -> dict:
    with _cache_lock:
        return {"enabled": CACHE_ENABLED, "entries": len(_cache)}


def clear_cache(delete_files: bool = False) -> int:
    """Clear the cache index (optionally deleting the audio files). Returns entries removed."""
    with _cache_lock:
        removed = len(_cache)
        if delete_files:
            for entry in _cache.values():
                try:
                    (OUTPUT_DIR / entry["filename"]).unlink(missing_ok=True)
                except OSError:
                    pass
        _cache.clear()
        _save_cache_index()
    return removed


def generate_cached(
    text: str,
    engine: str,
    voice: str | None,
    speaker_wav: str | None,
    filename: str | None = None,
) -> dict:
    """Return a result dict for `text`, serving the cache when possible.

    De-duplicates concurrent identical requests and serializes generation through the
    single worker thread. The returned dict adds a boolean ``cached`` flag. When cached,
    the previously generated file is served (any requested ``filename`` is ignored).
    """
    _ensure_worker()
    key = cache_key(text, engine, voice, speaker_wav)

    entry = _cache_lookup(key)
    if entry is not None:
        return {
            **entry,
            "path": str((OUTPUT_DIR / entry["filename"]).resolve()),
            "url": public_url(entry["filename"]),
            "cache_key": key,
            "cached": True,
        }

    with _jobs_lock:
        future = _inflight.get(key)
        if future is None:
            job = _Job(key, text, engine, voice, speaker_wav, filename)
            future = job.future
            _inflight[key] = future
            _job_queue.put(job)

    result = future.result()
    return {**result, "cache_key": key, "cached": False}
