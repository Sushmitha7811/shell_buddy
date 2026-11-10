"""Shell Buddy API server: loads the model once, then answers requests over HTTP."""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

import howto

state = {}
# One generation at a time: running two at once on a CPU just makes both slower
generate_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    state["tokenizer"], state["model"] = howto.load()   # runs ONCE when the server starts
    yield                                                # server handles requests here
    state.clear()                                        # runs once when the server stops


app = FastAPI(title="Shell Buddy", lifespan=lifespan)


class SuggestRequest(BaseModel):
    request: str = Field(min_length=1, max_length=300)


class SuggestResponse(BaseModel):
    command: str
    warnings: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/suggest", response_model=SuggestResponse)
def suggest(body: SuggestRequest):
    with generate_lock:
        cmd = howto.suggest(state["tokenizer"], state["model"], body.request)
    return SuggestResponse(command=cmd, warnings=howto.danger_warnings(cmd))
