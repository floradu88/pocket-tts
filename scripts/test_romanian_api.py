"""Smoke-test the Romanian TTS API service.

Launches the FastAPI service (pocket_tts.romanian.service:romanian_app) with uvicorn in a
subprocess, waits for /health, then exercises /engines, /voices and /tts for both engines,
and finally the mounted MCP endpoint (list tools + call a tool).

- Path A / secondary (pocket-tts Italian base) is fast.
- Path B / primary (XTTS-v2 Romanian) is slow on CPU (model load + synth take minutes).

Usage:
  python scripts/test_romanian_api.py
  python scripts/test_romanian_api.py --skip-xtts
  python scripts/test_romanian_api.py --mcp-only    # only exercise the MCP endpoint
  python scripts/test_romanian_api.py --port 8770
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tts_outputs" / "romanian_api"
TEST_TEXT = "Bună ziua! Acesta este un test al serviciului API în limba română."


def _wait_for_health(base_url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False


def _post_tts(base_url: str, engine: str, out_path: Path, timeout_s: float) -> bool:
    logger.info("POST /tts engine=%s (timeout %ds) ...", engine, int(timeout_s))
    try:
        resp = requests.post(
            f"{base_url}/tts",
            data={"text": TEST_TEXT, "engine": engine},
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        logger.error("engine=%s request failed: %s", engine, exc)
        return False

    if resp.status_code != 200:
        logger.error("engine=%s -> HTTP %d: %s", engine, resp.status_code, resp.text[:300])
        return False

    used = resp.headers.get("X-TTS-Engine-Used", "?")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    ok = len(resp.content) > 1000
    logger.info(
        "engine=%s -> OK (used=%s, %d bytes) -> %s",
        engine,
        used,
        len(resp.content),
        out_path,
    )
    return ok


async def _mcp_checks(base_url: str, generate: bool) -> bool:
    """List MCP tools and optionally generate speech through the MCP endpoint."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        logger.error("mcp not installed; install with: uv pip install -e '.[romanian-api]'")
        return False

    url = f"{base_url}/mcp"
    logger.info("Connecting to MCP endpoint: %s", url)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            logger.info("MCP tools: %s", names)
            expected = {
                "list_engines",
                "list_voices",
                "generate_romanian_speech",
                "generate_instagram_voiceover",
            }
            if not expected.issubset(set(names)):
                logger.error("Missing MCP tools: %s", expected - set(names))
                return False

            engines_result = await session.call_tool("list_engines", {})
            logger.info("MCP list_engines -> %s", _tool_text(engines_result))

            if generate:
                logger.info("MCP generate_romanian_speech (engine=pocket) [1st call] ...")
                gen = await session.call_tool(
                    "generate_romanian_speech",
                    {"text": TEST_TEXT, "engine": "pocket"},
                )
                first = _tool_json(gen)
                logger.info("MCP generate #1 -> %s", first)
                if not first or first.get("cached") is not False:
                    logger.error("First generation should not be cached: %s", first)
                    return False
                out = OUT_DIR / first["filename"]
                if not out.exists() or out.stat().st_size < 1000:
                    logger.error("MCP generation did not produce a valid WAV at %s", out)
                    return False

                logger.info("MCP generate_romanian_speech (engine=pocket) [2nd call] ...")
                gen2 = await session.call_tool(
                    "generate_romanian_speech",
                    {"text": TEST_TEXT, "engine": "pocket"},
                )
                second = _tool_json(gen2)
                logger.info("MCP generate #2 -> %s", second)
                if not second or second.get("cached") is not True:
                    logger.error("Second identical generation should be a cache hit: %s", second)
                    return False
                if second.get("filename") != first.get("filename"):
                    logger.error("Cache hit returned a different file: %s vs %s", second, first)
                    return False
    return True


def _tool_text(result) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        parts.append(getattr(item, "text", str(item)))
    structured = getattr(result, "structuredContent", None)
    if structured:
        parts.append(str(structured))
    return " ".join(parts)[:400]


def _tool_json(result) -> dict:
    """Extract a tool's structured result dict."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps non-dict returns under {"result": ...}; unwrap dict payloads.
        return structured.get("result", structured) if "result" in structured else structured
    import json as _json

    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            try:
                return _json.loads(text)
            except (ValueError, TypeError):
                continue
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Romanian TTS API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--skip-xtts", action="store_true", help="Only test the pocket engine.")
    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="Only exercise the MCP endpoint (list tools + list_engines), skip REST /tts.",
    )
    parser.add_argument(
        "--skip-mcp", action="store_true", help="Skip the MCP endpoint checks."
    )
    parser.add_argument(
        "--xtts-timeout",
        type=float,
        default=900.0,
        help="Timeout for XTTS requests (model load + synth are slow on CPU).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env["ROMANIAN_PRELOAD_SECONDARY"] = "1"
    env["ROMANIAN_PRELOAD_PRIMARY"] = "0"

    logger.info("Starting server: %s", base_url)
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "pocket_tts.romanian.service:romanian_app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )

    results: dict[str, bool] = {}
    try:
        if not _wait_for_health(base_url, timeout_s=120):
            logger.error("Server did not become healthy in time.")
            raise SystemExit(1)
        logger.info("Server healthy.")

        logger.info("GET /engines -> %s", requests.get(f"{base_url}/engines", timeout=10).json())
        logger.info("GET /voices  -> %s", requests.get(f"{base_url}/voices", timeout=10).json())

        if not args.mcp_only:
            results["pocket"] = _post_tts(
                base_url, "pocket", OUT_DIR / "api_pocket.wav", timeout_s=180
            )

            if not args.skip_xtts:
                results["xtts"] = _post_tts(
                    base_url, "xtts", OUT_DIR / "api_xtts.wav", timeout_s=args.xtts_timeout
                )
                results["auto"] = _post_tts(
                    base_url, "auto", OUT_DIR / "api_auto.wav", timeout_s=args.xtts_timeout
                )

        if not args.skip_mcp:
            results["mcp"] = asyncio.run(
                _mcp_checks(base_url, generate=args.mcp_only)
            )
    finally:
        logger.info("Stopping server...")
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()

    logger.info("---- Summary ----")
    for name, ok in results.items():
        logger.info("engine=%s: %s", name, "OK" if ok else "FAILED")

    mcp_ok = args.skip_mcp or results.get("mcp", False)
    if args.mcp_only:
        all_ok = mcp_ok
    else:
        all_ok = (
            results.get("pocket", False)
            and (args.skip_xtts or (results.get("xtts", False) and results.get("auto", False)))
            and mcp_ok
        )
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
