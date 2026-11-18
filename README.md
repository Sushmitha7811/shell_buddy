# Shell Buddy

A small AI model that turns plain English into Linux commands.

```
$ howto "find all files bigger than 1GB in my home folder"

  find ~ -size +1G

Run it? [y/N]
```

## Install (Linux)

You need Python 3. Shell Buddy runs on your own computer, so you don't need a GPU, an account or an internet connection after setup.

```bash
# 1. Install Ollama, which runs the model
curl -fsSL https://ollama.com/install.sh | sh

# 2. Download Shell Buddy (531 MB, one time)
ollama pull hf.co/SushmithaJujare/shell-buddy-GGUF

# 3. Install the howto command
mkdir -p ~/.local/bin
curl -fsSL -o ~/.local/bin/howto https://raw.githubusercontent.com/Sushmitha7811/shell_buddy/main/howto.py
chmod +x ~/.local/bin/howto
```

Open a new terminal and try:

```bash
howto "find all files bigger than 1GB in my home folder"
```

Model page: https://huggingface.co/SushmithaJujare/shell-buddy-GGUF

## How it was built

1. **Data:** ~10,800 English → bash pairs, cleaned from the public NL2Bash dataset (`clean_data.py`)
2. **Fine-tuning:** Qwen2.5-0.5B-Instruct trained with LoRA on a free Google Colab GPU (`shell_buddy_training.ipynb`)
3. **Evaluation:** scored on 100 unseen examples (`evaluate.py`, `eval_ollama.py`)
4. **Quantization:** converted to an 8-bit GGUF file (531 MB) and run with Ollama
5. **Tool:** `howto.py` asks the model, warns about risky commands, and only runs a command after you approve it

## Results

| Model | Right tool chosen | Exact match |
|---|---|---|
| Base Qwen 0.5B (not fine-tuned) | 62% | 5% |
| Qwen 7B (14× bigger, not fine-tuned) | 63% | 9% |
| **Shell Buddy (0.5B, fine-tuned, 8-bit)** | **91%** | **18%** |

## Notes

- Model files are not included in this repository (too large for GitHub).
- Shell Buddy can be wrong. Always read a command before running it.
- Known weakness: everyday commands like `pwd`, `df -h` and `du`, because most of the training data uses `find`.
