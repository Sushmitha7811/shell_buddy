# 2. Docker API server (FastAPI)

The model is loaded **once** by a small web server (`server.py`) that keeps it in memory and answers requests over HTTP. Docker packages the server, Python, PyTorch and the model into one image that runs the same on any machine.

```
curl / any program ──HTTP──► container: server.py (model already in memory) → command
```

| | |
|---|---|
| **Where the model runs** | Your laptop, inside a container |
| **Cost** | Free |
| **Speed** | ~1 s per request once started; ~1 minute to start |
| **Size** | 3.4 GB image, ~3 GB RAM while running |
| **Status** | Works. The current `howto.py` talks to Ollama, not to this server, so test it with `curl`. |

**When to choose it:** when you want to package a PyTorch model as an HTTP API. That's the most common way to deploy models in companies, and it's portable to any server that runs Docker.

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI app: loads the model at startup and serves `GET /health` and `POST /suggest`. Uses `howto.py` from the project root as a library. |
| `Dockerfile` | Build recipe: Python 3.10, CPU-only PyTorch, a non-root user, a health check |
| `requirements-server.txt` | Exact versions of transformers, fastapi and uvicorn |
| `../../.dockerignore` | Keeps the venv, data, adapters and GGUF out of the build (it must sit at the build root) |

## What you need

- Docker
- `shell-buddy-merged/` at the project root. It isn't published, so build it with [training/](../../training/).
- ~4 GB of free disk space

## Build and run

Build **from the project root**, because the image needs `howto.py` and `shell-buddy-merged/` from there. `-f` says where the Dockerfile is.

```bash
docker build -f deploy/2-docker-api/Dockerfile -t shell-buddy:v2 .
docker run -d --name shell-buddy -p 127.0.0.1:8765:8765 shell-buddy:v2
docker logs -f shell-buddy        # wait for "Application startup complete", then Ctrl+C
docker ps                         # shows "(healthy)" after ~1 minute
```

`-p 127.0.0.1:8765:8765` makes it reachable **only from your own laptop**. Don't use `-p 8765:8765` or `--host 0.0.0.0`: this server has no API key, so anyone on your network could use it.

## Test

```bash
curl -s http://127.0.0.1:8765/health
curl -s -X POST http://127.0.0.1:8765/suggest \
  -H "Content-Type: application/json" \
  -d '{"request": "find all png files in this folder"}'
```

FastAPI also generates a test page at http://127.0.0.1:8765/docs.

## Manage

```bash
docker stop shell-buddy       # stop (frees ~3 GB RAM)
docker start shell-buddy      # start again
docker rm -f shell-buddy      # delete the container (the image stays)
docker rmi shell-buddy:v2     # delete the image (frees ~3.4 GB)
```

## Problems I hit

| Symptom | Cause | Fix |
|---|---|---|
| `flit_core` not found during build | The base image ships an old pip | `RUN pip install --upgrade pip` first |
| `FileNotFoundError` for a file that exists | Files copied as root; the app runs as `appuser` | `useradd` **before** `COPY --chown=appuser:appuser` |
| Port 8000 busy | Another app on the laptop | Use 8765 |

## Why I moved on

3.4 GB and ~3 GB of RAM is a lot for a 0.5B model. Converting to an 8-bit GGUF and running it with Ollama cut that to 531 MB and ~600 MB of RAM ([approach 3](../3-ollama-local/)).

Details: [SHELL_BUDDY_GUIDE.md, Lessons 7–8](../../SHELL_BUDDY_GUIDE.md#lesson-7-deploy-as-an-api-server).
