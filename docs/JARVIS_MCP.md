# Romanian TTS as a Jarvis MCP service

This guide shows how to run the Romanian TTS engine as a Dockerized API + MCP server and
wire it into a Jarvis agent so it can generate Romanian voiceovers (e.g. for Instagram
Reels) as a tool call.

The container exposes two things on the same port:

- a **REST API** (`/tts`, `/health`, `/engines`, `/voices`, `/cache`, `/outputs/<file>`)
- an **MCP server** at **`/mcp`** (streamable HTTP) that Jarvis connects to

Generation is **cached and de-duplicated**: identical requests are served from disk and are
never synthesized twice, and all generation runs through a single serialized queue (the
models are not thread-safe). See [Caching & queue](#caching--queue) below.

Engines:

| Engine   | Role       | Quality                          | Speed (CPU)         |
| -------- | ---------- | -------------------------------- | ------------------- |
| `xtts`   | primary    | native Romanian pronunciation    | slow (~real-time)   |
| `pocket` | secondary  | fast draft, Italian accent       | fast                |
| `auto`   | —          | tries `xtts`, falls back to `pocket` | —              |

## 1. Build and start the service

From the repo root (PowerShell):

```powershell
# Build the image
./deploy/build.ps1

# Start via docker compose (creates .env from .env.example on first run)
./deploy/start.ps1 -Build
```

Or run a single container without compose:

```powershell
./deploy/run.ps1 -Detach -Port 8001
```

First startup downloads the XTTS model (hundreds of MB) and the pocket-tts weights; this
can take several minutes. The model caches are stored in Docker volumes so subsequent
starts are fast.

Verify it's up:

```powershell
curl http://localhost:8001/health
curl http://localhost:8001/engines
```

Generated audio is written to `tts_outputs/romanian_api/` on the host (bind-mounted to
`/data/outputs` in the container) and is downloadable at `http://localhost:8001/outputs/<file>`.

## 2. Configuration (.env)

Copy `.env.example` to `.env` and adjust. Key settings:

| Variable                     | Default                  | Meaning                                             |
| ---------------------------- | ------------------------ | --------------------------------------------------- |
| `ROMANIAN_PORT`              | `8001`                   | Host port (container always listens on 8000)        |
| `ROMANIAN_DEFAULT_ENGINE`    | `auto`                   | `auto` \| `xtts` \| `pocket`                         |
| `ROMANIAN_MAX_CHARS`         | `1000`                   | Max characters per request                          |
| `ROMANIAN_PRELOAD_PRIMARY`   | `1`                      | Load XTTS at startup (else on first use)            |
| `ROMANIAN_PRELOAD_SECONDARY` | `1`                      | Load pocket-tts at startup                          |
| `ROMANIAN_CACHE`             | `1`                      | Cache + de-duplicate generations (`0` to disable)   |
| `ROMANIAN_PUBLIC_BASE_URL`   | `http://localhost:8001`  | Base URL used in MCP tool result download links     |

> If Jarvis runs in another container on the same Docker network, set
> `ROMANIAN_PUBLIC_BASE_URL=http://romanian-tts:8000` so the URLs it receives are reachable.

## 3. The MCP endpoint

The MCP server speaks **streamable HTTP** at:

```
http://<host>:<port>/mcp
```

### Tools exposed

| Tool                            | Purpose                                                            |
| ------------------------------- | ----------------------------------------------------------------- |
| `list_engines`                  | Report engines, load status, cache stats and queue depth          |
| `list_voices`                   | List XTTS speaker ids                                              |
| `generate_romanian_speech`      | Synthesize Romanian text → saved WAV (`text`, `engine`, `voice`, `filename`) |
| `generate_instagram_voiceover`  | Same, defaulting to the `xtts` engine for best quality (`script`, `voice`, `engine`, `filename`) |

Each generation tool returns:

```json
{
  "filename": "ro_c8b73a6cc655a0bd.wav",
  "path": "/data/outputs/ro_c8b73a6cc655a0bd.wav",
  "url": "http://localhost:8001/outputs/ro_c8b73a6cc655a0bd.wav",
  "engine_used": "xtts",
  "sample_rate": 24000,
  "duration_s": 6.42,
  "cache_key": "c8b73a6cc655a0bd",
  "cached": false
}
```

- `cached` is `true` when the audio was served from cache (no regeneration).
- `cache_key` is the content hash used for de-duplication.
- The default filename is derived from `cache_key`, so identical requests always map to the
  same file. A `filename` you pass is only applied on the first (cache-miss) generation.

Jarvis can either download the audio from `url` or read it directly from the shared
`tts_outputs/romanian_api/` folder using `path`.

## 4. Register the tool with Jarvis

Add the MCP server to Jarvis's MCP configuration. The exact file depends on your Jarvis
setup, but the shape is the standard MCP client config:

```json
{
  "mcpServers": {
    "romanian-tts": {
      "type": "http",
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

If your Jarvis MCP client uses the `url`/`transport` style instead:

```json
{
  "mcpServers": {
    "romanian-tts": {
      "transport": "streamable-http",
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

After Jarvis reconnects, the four tools above appear in its toolset. A typical Instagram
content flow:

1. Jarvis writes the Romanian caption/script.
2. Jarvis calls `generate_instagram_voiceover(script="...", voice="...")`.
3. Jarvis takes the returned `url`/`path` and muxes the audio into the video.

## 5. Caching & queue

The service keeps Jarvis fast and avoids wasted CPU:

- **Content cache** — every generation is keyed by a hash of `(text, engine, voice,
  reference clip)`. If the same request comes in again, the previously produced WAV is
  served straight from disk (`cached: true`) instead of being regenerated. The cache index
  is stored at `<output dir>/cache_index.json` and survives restarts (the output dir is a
  bind mount, so the cache persists on the host).
- **In-flight de-duplication** — if two identical requests arrive at once, only one is
  generated; the second waits for and receives the same result.
- **Serialized queue** — all generations run through a single worker thread, so concurrent
  requests are queued and processed one at a time (the models are not thread-safe). The
  current backlog is reported as `queue_depth` by `list_engines` and `GET /engines`.

Inspect and manage the cache over REST:

```powershell
# Cache stats + current queue depth
curl http://localhost:8001/cache

# Clear the cache index (keep the audio files)
curl -X DELETE http://localhost:8001/cache

# Clear the index AND delete the generated files
curl -X DELETE "http://localhost:8001/cache?delete_files=true"
```

REST responses also carry cache info in headers: `X-TTS-Cache: hit|miss`,
`X-TTS-Engine-Used`, and `X-TTS-Filename`.

## 6. Quick manual checks

REST (bypassing MCP):

```powershell
# pocket (fast) engine
curl -X POST http://localhost:8001/tts `
  -F "text=Salut, acesta este un test." `
  -F "engine=pocket" `
  --output test_pocket.wav

# xtts (native) engine
curl -X POST http://localhost:8001/tts `
  -F "text=Salut, acesta este un test." `
  -F "engine=xtts" `
  --output test_xtts.wav
```

MCP tool listing with the bundled smoke test (requires the `romanian-api` extra installed
locally):

```powershell
uv run python scripts/test_romanian_api.py --mcp-only
```

## 7. Troubleshooting

- **First request is slow / times out**: the model is still downloading or loading. Watch
  `docker compose logs -f romanian-tts`. Set `ROMANIAN_PRELOAD_PRIMARY=1` to front-load it
  at startup instead of on the first call.
- **XTTS fails, pocket works**: XTTS needs its model download and ffmpeg; the image ships
  ffmpeg, but if XTTS still fails, `auto` will fall back to `pocket`. Check logs for the
  underlying error.
- **URLs not reachable from Jarvis**: set `ROMANIAN_PUBLIC_BASE_URL` to an address Jarvis
  can actually reach (service name on a shared network, or a public/reverse-proxied URL).
- **License prompt**: Coqui XTTS is non-commercial; `COQUI_TOS_AGREED=1` is set in the
  image/compose to accept it non-interactively.
- **Stale/unwanted cache hits**: identical text returns the same audio by design. To force
  a fresh render, change the text/voice/engine, clear the cache (`DELETE /cache`), or set
  `ROMANIAN_CACHE=0`.
