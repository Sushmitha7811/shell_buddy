"""Public front door for Shell Buddy: checks API keys, then asks Ollama."""
import json
import os
import secrets
import time
import urllib.request
from collections import defaultdict, deque

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "shell-buddy"
SYSTEM_PROMPT = ("You are Shell Buddy. Convert the user's request into a single Linux bash "
                 "command. Reply with only the command, no explanation and no markdown.")
# Comma-separated keys, read from the environment, never written in code.
API_KEYS = [k.strip() for k in os.environ.get("SHELL_BUDDY_KEYS", "").split(",") if k.strip()]
RATE_LIMIT = 20  # requests per key per minute

# docs_url=None etc. hides the auto-generated docs pages from strangers.
app = FastAPI(title="Shell Buddy", docs_url=None, redoc_url=None, openapi_url=None)
recent_requests = defaultdict(deque)


class SuggestRequest(BaseModel):
    request: str = Field(min_length=1, max_length=300)


def check_key(given: str) -> str:
    for key in API_KEYS:
        # compare_digest takes the same time whether 1 or 40 characters match,
        # so attackers can't guess the key letter by letter from response times.
        if secrets.compare_digest(given.encode(), key.encode()):
            return key
    raise HTTPException(status_code=401, detail="invalid API key")


def check_rate(key: str) -> None:
    now = time.monotonic()
    times = recent_requests[key]
    while times and now - times[0] > 60:
        times.popleft()
    if len(times) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="too many requests, wait a minute")
    times.append(now)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/suggest")
def suggest(body: SuggestRequest, x_api_key: str = Header(default="")):
    key = check_key(x_api_key)
    check_rate(key)
    payload = json.dumps({
        "model": MODEL, "system": SYSTEM_PROMPT, "prompt": body.request,
        "stream": False, "options": {"temperature": 0, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = json.loads(resp.read())["response"]
    except Exception:
        raise HTTPException(status_code=503, detail="model unavailable") from None
    lines = text.strip().splitlines()
    return {"command": lines[0].strip() if lines else ""}
