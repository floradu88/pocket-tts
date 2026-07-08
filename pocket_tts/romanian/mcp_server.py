"""MCP server exposing Romanian TTS tools for agents (e.g. Jarvis).

Runs in-process inside the same container as the REST API and reuses the shared engine
singletons from `runtime`. It is exposed over streamable HTTP so a remote MCP client can
connect at `http://<host>:<port>/mcp`.

Tools return the saved file path and a public download URL rather than inline audio, so
large voiceovers are handled efficiently and can be picked up by Instagram tooling from the
shared output folder.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from pocket_tts.romanian import runtime

logger = logging.getLogger(__name__)

# stateless_http keeps each tool call independent (no persistent SSE session needed), which
# makes mounting into FastAPI simple. streamable_http_path="/" so that mounting the app at
# "/mcp" exposes the endpoint exactly at "/mcp".
mcp = FastMCP(
    "romanian-tts",
    instructions=(
        "Romanian text-to-speech for content generation. Use generate_romanian_speech or "
        "generate_instagram_voiceover to synthesize Romanian audio. The 'xtts' engine gives "
        "native pronunciation (slower); 'pocket' is a fast Italian-accented draft; 'auto' "
        "prefers xtts and falls back to pocket."
    ),
    stateless_http=True,
    streamable_http_path="/",
)


@mcp.tool()
def list_engines() -> dict:
    """List Romanian TTS engines, load status, cache stats and queue depth."""
    return {
        "default_engine": runtime.DEFAULT_ENGINE,
        "cache": runtime.cache_stats(),
        "queue_depth": runtime.queue_depth(),
        "engines": {
            name: {
                "role": "primary" if name == "xtts" else "secondary",
                "loaded": engine.is_loaded,
            }
            for name, engine in runtime.ENGINES.items()
        },
    }


@mcp.tool()
def list_voices() -> dict:
    """List built-in XTTS Romanian speaker ids (available once the XTTS model is loaded)."""
    speakers = runtime.xtts_engine.available_speakers() if runtime.xtts_engine.is_loaded else []
    return {
        "xtts_speakers": speakers,
        "xtts_loaded": runtime.xtts_engine.is_loaded,
        "note": (
            "If the list is empty, speakers appear after the first XTTS generation. For the "
            "pocket engine, use a predefined voice name such as 'giovanni'."
        ),
    }


@mcp.tool()
def generate_romanian_speech(
    text: str,
    engine: str = "auto",
    voice: str | None = None,
    filename: str | None = None,
) -> dict:
    """Generate Romanian speech and save it as a WAV file.

    Args:
        text: Romanian text to synthesize.
        engine: "auto" (xtts then pocket fallback), "xtts" (native), or "pocket" (fast draft).
        voice: XTTS speaker id or pocket-tts voice name (optional; a default is used).
        filename: Optional output filename (a unique name is generated if omitted).

    Identical requests are served from cache (see the ``cached`` field) and are never
    generated twice. Returns filename, path, url, engine_used, sample_rate, duration_s,
    cache_key and cached.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text cannot be empty")
    if len(text) > runtime.MAX_CHARS:
        raise ValueError(f"text too long ({len(text)} > {runtime.MAX_CHARS} chars)")
    engine = (engine or "auto").lower()
    if engine not in runtime.VALID_ENGINES:
        raise ValueError("engine must be auto, xtts, or pocket")

    return runtime.generate_cached(text, engine, voice, None, filename)


@mcp.tool()
def generate_instagram_voiceover(
    script: str,
    voice: str | None = None,
    engine: str = "xtts",
    filename: str | None = None,
) -> dict:
    """Generate a native-quality Romanian voiceover for Instagram/Reels content.

    Defaults to the XTTS engine for the best pronunciation. Same return shape as
    generate_romanian_speech.
    """
    return generate_romanian_speech(script, engine=engine, voice=voice, filename=filename)


# ASGI app for mounting into the FastAPI service (streamable HTTP transport).
mcp_app = mcp.streamable_http_app()
