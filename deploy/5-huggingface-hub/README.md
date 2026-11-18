# 5. Hugging Face Hub + Ollama ⭐ recommended

The model file is published on the **Hugging Face Hub**, a free public home for model files. Ollama can download from the Hub directly, so every user runs Shell Buddy **on their own laptop**: no server to pay for, no API keys, and it works offline.

```
Hugging Face Hub ──one-time download (531 MB)──► user's Ollama ◄── howto.py (same laptop)
```

| | |
|---|---|
| **Where the model runs** | Each user's own laptop |
| **Cost** | Free, for you and for users |
| **Speed** | ~1 s per request |
| **Status** | ⭐ **In use.** This is what the [main install guide](../../README.md#install-linux) sets up. |
| **Model page** | https://huggingface.co/SushmithaJujare/shell-buddy-GGUF |

**When to choose it:** for almost everyone. The only cost is that each user installs Ollama and downloads 531 MB once.

## Using it

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hf.co/SushmithaJujare/shell-buddy-GGUF
ollama run hf.co/SushmithaJujare/shell-buddy-GGUF "show free memory"
```

`howto.py` uses this model name by default. Full setup: [main README](../../README.md#install-linux).

## Files

A `.gguf` file contains only the model's weights. The **settings** in a Modelfile (system prompt, chat format, temperature) aren't inside it. When Ollama downloads from the Hub, it reads them from small files in the repo instead:

| File in `hub_files/` | Same as this Modelfile line | Purpose |
|---|---|---|
| `system` | `SYSTEM` | The Shell Buddy system prompt. Without it, Ollama uses Qwen's default ("You are Qwen…") and answers get worse. |
| `template` | `TEMPLATE` | The chat format the model was trained on (Go template syntax, not Jinja) |
| `params` | `PARAMETER` | `temperature 0`, `num_predict 80` and the stop token, written as JSON |
| `README.md` | — | The model card: the Hub page visitors see |

To check that Ollama picked them up:
```bash
ollama show --modelfile hf.co/SushmithaJujare/shell-buddy-GGUF
```

## Publishing (or updating) the model

```bash
pip install -U huggingface_hub
hf auth login                      # needs an access token with "Write" permission
hf upload SushmithaJujare/shell-buddy-GGUF shell-buddy-q8_0.gguf
hf upload SushmithaJujare/shell-buddy-GGUF deploy/5-huggingface-hub/hub_files/ .
```

The `.` at the end puts the files at the top level of the repo, next to the `.gguf`. The repo is created automatically on the first upload.

**Check it like a new user would.** Both commands should give the same answer:
```bash
ollama run hf.co/SushmithaJujare/shell-buddy-GGUF "show disk usage of this folder"
ollama run shell-buddy "show disk usage of this folder"        # your local build (approach 3)
```

## Notes

- The newline at the end of `template` was dropped on upload, so the format ends at `assistant` without one. Answers still match the local build. If they ever differ, check this first.
- **Alternative registry:** `ollama push <username>/shell-buddy` publishes to ollama.com instead, for users who prefer `ollama pull <username>/shell-buddy`.
