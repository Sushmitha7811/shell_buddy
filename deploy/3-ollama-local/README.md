# 3. Ollama, local build

The merged model is converted to an 8-bit **GGUF** file (531 MB) and imported into **Ollama**, a program that runs GGUF models quickly on ordinary hardware and serves an API on port 11434. `howto.py` asks Ollama over HTTP, so it never loads PyTorch.

| | |
|---|---|
| **Where the model runs** | Your laptop (CPU, or a small GPU: it ran on a 2 GB MX230) |
| **Cost** | Free |
| **Speed** | ~1 s per request; ~600 MB RAM |
| **Accuracy** | 91% right program vs 92% before rounding to 8-bit |
| **Status** | Works. Use this one if you build the GGUF yourself. |

**When to choose it:** when you've trained or converted your own model and want to run that file. If you just want to use Shell Buddy, [approach 5](../5-huggingface-hub/) downloads the same model without building anything.

## Files

| File | Purpose |
|---|---|
| `Modelfile` | Tells Ollama which GGUF file to load (`FROM ../../shell-buddy-q8_0.gguf`), the chat format it was trained on, the Shell Buddy system prompt and `temperature 0` |

The system prompt and chat format must be **exactly** what the model was trained with, or the answers get worse.

## What you need

- [Ollama](https://ollama.com):
  ```bash
  curl -fsSL https://ollama.com/install.sh -o ollama-install.sh
  less ollama-install.sh      # read it before running it
  sh ollama-install.sh
  ```
- `shell-buddy-q8_0.gguf` at the project root (build it with step 8 in [training/](../../training/))

## Import and run

From the project root:

```bash
ollama create shell-buddy -f deploy/3-ollama-local/Modelfile
ollama list                                          # shell-buddy:latest  531 MB
ollama run shell-buddy "find all files bigger than 1GB in my home folder"
```

`howto.py` uses the Hugging Face model name by default, so tell it to use this local one:

```bash
SHELL_BUDDY_MODEL=shell-buddy python howto.py "find all png files in this folder"
```

`ollama ps` shows what's loaded and whether it's on the CPU or GPU. Ollama unloads a model 5 minutes after its last use.

## Problems I hit

Ollama 0.40's `--quantize` option can't quantize a GGUF file, and for other inputs it only offers Apple MLX formats. **Quantize with llama.cpp first** (`--outtype q8_0`), then import the finished file without `--quantize`. For a 0.5B model, 8-bit is safer than 4-bit, which loses more accuracy on small models.

## What this doesn't solve

It only runs on your own laptop. To let others use it, I first rented a cloud server ([approach 4](../4-aws-ec2/)), and later published the model file instead ([approach 5](../5-huggingface-hub/)).

Details: [SHELL_BUDDY_GUIDE.md, Lesson 9](../../SHELL_BUDDY_GUIDE.md#lesson-9-quantization-with-ollama).
