# Shell Buddy

A small AI model that turns plain English into Linux commands.

```
$ howto "find all files bigger than 1GB in my home folder"

  find ~ -size +1G

Run it? [y/N]
```

Shell Buddy is Qwen2.5-0.5B fine-tuned on ~10,800 English → bash examples. It runs on an ordinary laptop CPU, and it never runs a command until you approve it.

## Install (Linux)

You need Python 3. Shell Buddy runs on your own computer: no GPU, account or API key, and no internet connection after setup.

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

### Troubleshooting

| Message | Fix |
|---|---|
| `howto: command not found` | `~/.local/bin` isn't in your PATH yet (Ubuntu adds it only if the folder existed when you logged in). Run `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc` |
| `The Shell Buddy model isn't downloaded yet` | Run step 2 again |
| `Can't reach Ollama` | Start it with `sudo systemctl start ollama` |

Model page: https://huggingface.co/SushmithaJujare/shell-buddy-GGUF

## Six ways I ran it

This project tried several ways of running and sharing the model. Each one has its own folder with the files and steps, so you can follow whichever fits you.

| # | Approach | Where the model runs | Cost | Status |
|---|---|---|---|---|
| 1 | [Local Python](deploy/1-local-python/) | Your laptop, with PyTorch | Free | Works, but slow (~8 s per request) |
| 2 | [Docker API server](deploy/2-docker-api/) | Your laptop, in a container | Free | Works; 3.4 GB image |
| 3 | [Ollama, local build](deploy/3-ollama-local/) | Your laptop, from your own GGUF file | Free | Works (~1 s per request) |
| 4 | [AWS EC2 public server](deploy/4-aws-ec2/) | A rented cloud server, behind HTTPS + API keys | ~$0.10/hour from credits | Shut down (cost) |
| 5 | [Hugging Face Hub + Ollama](deploy/5-huggingface-hub/) | Each user's own laptop | Free |  **Recommended**, what the install above uses |
| 6 | [Hugging Face Spaces](deploy/6-hf-spaces-tried/) | Hugging Face's servers | Paid | Not possible on the free plan |

**Which one should I use?**
- **Just want to use Shell Buddy:** the [Install](#install-linux) section above (approach 5).
- **Want a server other people can call without installing anything:** approach 4 (AWS). It works, but costs money while running.
- **Learning how deployment works:** read them in order. Each one fixes a problem with the one before it.

## Project layout

```
howto.py           the command-line tool (asks the model, checks for danger, asks before running)
test_howto.py      tests for the danger warnings
training/          how the model was built: data cleaning, fine-tuning, evaluation, merging
deploy/            one folder per way of running it (table above)
SHELL_BUDDY_GUIDE.md   the full step-by-step guide, every lesson and every problem I hit
```

## How it was built

1. **Data:** ~10,800 English → bash pairs, cleaned from the public NL2Bash dataset
2. **Fine-tuning:** LoRA on a free Google Colab GPU
3. **Evaluation:** scored on 100 unseen examples
4. **Merging and quantization:** the adapter was merged into the base model and converted to an 8-bit GGUF file (531 MB)
5. **Tool:** `howto.py` asks the model, warns about risky commands, and runs a command only after you approve it

Details and scripts: [training/](training/).

## Results

| Model | Right tool chosen | Exact match |
|---|---|---|
| Base Qwen 0.5B (not fine-tuned) | 62% | 5% |
| Qwen 7B (14× bigger, not fine-tuned) | 63% | 9% |
| **Shell Buddy (0.5B, fine-tuned, 8-bit)** | **91%** | **18%** |

A small model fine-tuned for one task beats a 14× bigger general model at that task.

## Safety

- `howto` **never runs a command without asking.** The default answer is "no", and risky commands (deleting files, `sudo`, `dd`, `kill`, …) need you to type the full word `yes`.
- Commands that only make sense in your own shell (`cd`, `export`, …) are shown for you to copy, not run.
- If the model answers with something that isn't a real program, `howto` refuses to run it.

## Notes

- Model files are too big for GitHub, so they live on [Hugging Face](https://huggingface.co/SushmithaJujare/shell-buddy-GGUF).
- Shell Buddy can be wrong. Always read a command before running it.
- Known weakness: everyday commands like `pwd`, `df -h` and `du`, because 63% of the training data uses `find`.
- `SHELL_BUDDY_GUIDE.md` was written before the files moved into `training/` and `deploy/`, so its commands use the old paths (for example `python clean_data.py` instead of `python training/clean_data.py`).
