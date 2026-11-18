# 1. Local Python (PyTorch)

The first and simplest way: `howto.py` loads the merged model straight into Python with PyTorch and transformers, asks it for a command, and exits.

| | |
|---|---|
| **Where the model runs** | Your laptop's CPU |
| **Cost** | Free |
| **Speed** | ~8–10 s per request, almost all of it loading PyTorch and the 1 GB model **every time** |
| **Status** | Works. Used now only as a fallback when Ollama isn't installed. |

**When to choose it:** for experiments, when you already have the Python training environment and the `shell-buddy-merged/` folder. For everyday use, approach 3 or 5 is ~10× faster.

## What you need

- The merged model in `shell-buddy-merged/` at the project root. **It isn't published**, so build it with [training/](../../training/) (`merge_model.py`).
- Python with `torch` (CPU version) and `transformers`:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers
  ```

There are no files in this folder. The code is the `load()` and `suggest()` functions in [howto.py](../../howto.py).

## How to run it

`howto.py` tries Ollama first. It uses the local model only when Ollama can't be reached **and** `shell-buddy-merged/` exists. To force the local path, point it at a port where nothing is listening:

```bash
SHELL_BUDDY_URL=http://127.0.0.1:1 python howto.py "find all png files in this folder"
```

You'll see `(Ollama not reachable, loading the model locally...)`, then the suggestion after a few seconds.

## Why I moved on

Each request started Python, imported PyTorch (~2 s), loaded the model (~5 s), generated (~1 s) and then threw everything away. A **server** that loads the model once fixes that, which led to [approach 2](../2-docker-api/).

Details: [SHELL_BUDDY_GUIDE.md, Lesson 6](../../SHELL_BUDDY_GUIDE.md#lesson-6-merge-the-model-and-build-howto).
