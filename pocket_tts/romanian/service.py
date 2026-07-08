"""FastAPI service exposing Romanian TTS with a primary and secondary engine, plus an
in-process MCP server (for agents such as Jarvis) mounted at /mcp.

Engines:
- primary   = XTTS-v2 Romanian fine-tune (native pronunciation, slower on CPU)
- secondary = pocket-tts Italian base (fast, Italian-accented)

Endpoints:
- GET  /health         liveness
- GET  /engines        engine load status
- GET  /voices         XTTS speaker catalog
- POST /tts            generate speech (engine=auto|xtts|pocket)
- GET  /outputs/<file> download generated audio
- /mcp                 MCP streamable-HTTP endpoint for agents

Configuration is read from environment variables (see runtime.py and README/JARVIS_MCP.md).
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from pocket_tts.romanian import runtime
from pocket_tts.romanian.mcp_server import mcp, mcp_app

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.preload()
    # Run the MCP session manager for the mounted streamable-HTTP app.
    async with mcp.session_manager.run():
        yield


romanian_app = FastAPI(
    title="Romanian TTS API",
    description="Romanian TTS with XTTS-v2 (primary) and pocket-tts (secondary), plus MCP.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve generated audio for download (shared output directory).
runtime.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
romanian_app.mount("/outputs", StaticFiles(directory=str(runtime.OUTPUT_DIR)), name="outputs")

# Mount the MCP server so agents can connect at /mcp.
romanian_app.mount("/mcp", mcp_app)


@romanian_app.get("/health")
def health():
    return {"status": "healthy"}


@romanian_app.get("/engines")
def engines():
    return {
        "default_engine": runtime.DEFAULT_ENGINE,
        "cache": runtime.cache_stats(),
        "queue_depth": runtime.queue_depth(),
        "engines": {
            name: {
                "name": name,
                "role": "primary" if name == "xtts" else "secondary",
                "loaded": engine.is_loaded,
            }
            for name, engine in runtime.ENGINES.items()
        },
    }


@romanian_app.get("/cache")
def cache_status():
    return {**runtime.cache_stats(), "queue_depth": runtime.queue_depth()}


@romanian_app.delete("/cache")
def cache_clear(delete_files: bool = False):
    removed = runtime.clear_cache(delete_files=delete_files)
    return {"removed": removed, "deleted_files": delete_files}


@romanian_app.get("/voices")
def voices():
    xtts_speakers = (
        runtime.xtts_engine.available_speakers() if runtime.xtts_engine.is_loaded else []
    )
    return {
        "xtts": {
            "loaded": runtime.xtts_engine.is_loaded,
            "speakers": xtts_speakers,
            "note": "If not loaded, speakers appear after the first XTTS request. "
            "You may also upload a reference clip via speaker_wav.",
        },
        "pocket": {
            "note": "Use a predefined pocket-tts voice name (e.g. giovanni) or upload a clip."
        },
    }


@romanian_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    engine: str = Form(None),
    voice: str | None = Form(None),
    speaker_wav: UploadFile | None = File(None),
):
    """Generate Romanian speech and return WAV bytes (also see the MCP tools for agents)."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > runtime.MAX_CHARS:
        raise HTTPException(
            status_code=400, detail=f"Text too long ({len(text)} > {runtime.MAX_CHARS} chars)"
        )

    requested = (engine or runtime.DEFAULT_ENGINE).lower()
    if requested not in runtime.VALID_ENGINES:
        raise HTTPException(status_code=400, detail="engine must be auto, xtts, or pocket")

    temp_path: str | None = None
    if speaker_wav is not None:
        suffix = Path(speaker_wav.filename).suffix if speaker_wav.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(speaker_wav.file.read())
            temp_path = temp_file.name

    try:
        try:
            result = runtime.generate_cached(text, requested, voice, temp_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc))

        audio = Path(result["path"]).read_bytes()
        return Response(
            content=audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={result['filename']}",
                "X-TTS-Engine-Used": result["engine_used"],
                "X-TTS-Cache": "hit" if result["cached"] else "miss",
                "X-TTS-Filename": result["filename"],
            },
        )
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = os.environ.get("ROMANIAN_HOST", "0.0.0.0")
    port = int(os.environ.get("ROMANIAN_PORT", "8000"))
    uvicorn.run(romanian_app, host=host, port=port)


if __name__ == "__main__":
    main()
