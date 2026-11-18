# Training: how Shell Buddy was built

From a general chat model to a model that answers with a single bash command.

```
NL2Bash dataset → clean → fine-tune (LoRA, Colab) → evaluate → merge → convert to GGUF
```

**Run every script from the project root**, not from inside `training/`. The scripts read and write `data/` and the model folders using paths relative to where you run them:

```bash
cd shell_buddy
python training/clean_data.py
```

## Setup

You need ~15 GB of free disk space. A GPU isn't needed on your own machine, because training runs on Google Colab.

```bash
python3 -m venv .mb
source .mb/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate peft trl
```

## The scripts, in order

| Step | File | What it does | Output |
|---|---|---|---|
| 1 | `explore_data.py` | First look at the raw dataset: random examples, most common programs, duplicates | Printed report |
| 2 | `run_base.py` | Asks the **untrained** base model, to see how it fails (markdown, explanations, wrong flags) | Printed answers |
| 3 | `clean_data.py` | Removes duplicates, broken commands, curly quotes and multi-line scripts, then splits into train and test | `data/train.jsonl`, `data/test.jsonl`, `data/review_sample.txt` |
| 4 | `inspect_data.py` | Shows one example in the chat format and measures token lengths (used to pick `max_length`) | Printed report |
| 5 | `shell_buddy_training.ipynb` | LoRA fine-tuning on Colab's free T4 GPU (upload `data/train.jsonl` only) | `shell-buddy-lora-v2/` adapter (~35 MB) |
| 6 | `evaluate.py` | Scores the adapter on unseen test examples | Scores + `data/eval_results_*.jsonl` |
| 7 | `merge_model.py` | Bakes the adapter into the base model, giving one finished model | `shell-buddy-merged/` (~1 GB) |
| 8 | llama.cpp converter | Converts to GGUF and rounds to 8-bit (see below) | `shell-buddy-q8_0.gguf` (531 MB) |
| 9 | `eval_ollama.py` | Scores the 8-bit model through Ollama, to check that rounding didn't hurt | Scores |

```bash
python training/explore_data.py
python training/clean_data.py
python training/inspect_data.py
# → train on Colab with shell_buddy_training.ipynb, download the adapter folder
python training/evaluate.py 100 shell-buddy-lora-v2    # ~5–10 min on CPU
python training/merge_model.py
```

**Step 8: convert to GGUF.** Install only llama.cpp's `gguf-py` package. Its full requirements file pins its own torch and transformers versions and can break your environment.

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ../llama.cpp
pip install ../llama.cpp/gguf-py sentencepiece
python ../llama.cpp/convert_hf_to_gguf.py shell-buddy-merged --outfile shell-buddy-q8_0.gguf --outtype q8_0
```

Then import it into Ollama ([deploy/3-ollama-local](../deploy/3-ollama-local/)) and score it:

```bash
python training/eval_ollama.py 100 shell-buddy
```

## Training settings

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA, `r=16`, `lora_alpha=32` (~1.75% of weights trained) |
| Epochs / batch size / learning rate | 2 / 16 / 2e-4 |
| `max_length` | 256 (longest example is 188 tokens) |
| Precision | `fp16=True, bf16=False` (the T4 doesn't support bf16) |
| Loss | On the answer (the command) only, not the prompt |

## Results (100 unseen test examples)

| Model | Clean format | Valid syntax | Right program | Exact match |
|---|---|---|---|---|
| Base Qwen 0.5B | 98% | 99% | 62% | 5% |
| Qwen 7B (not trained) | 99% | 100% | 63% | 9% |
| Shell Buddy v2, 16-bit | 100% | 100% | 92% | 19% |
| **Shell Buddy v2, 8-bit GGUF** | **100%** | **99%** | **91%** | **18%** |

"Exact match" undercounts: when I read the mismatches by hand, most of them were different commands that also work.

## Lessons learned

- **v1 → v2:** the model started printing curly quotes (`“ ”`), which bash doesn't understand. They came from the training data. Fixing the data and retraining produced v2.
- **Only compare models on the same test set.** Re-running `clean_data.py` once changed the test rows, and the base model's score moved from 53% to 62% without the model changing.
- **Data imbalance:** 63% of the examples use `find`, so everyday commands (`pwd`, `df -h`, `du`) are weak. That's the planned fix for v3.
- **Colab:** if PEFT fails with a `torchao` version error, run `pip uninstall -y torchao` and restart the session.

`data/` isn't in this repo because the dataset's license is unclear. Recreate it with `clean_data.py`, which downloads `AnishJoshi/nl2bash-custom` from Hugging Face.

Full walk-through: [SHELL_BUDDY_GUIDE.md](../SHELL_BUDDY_GUIDE.md), Lessons 1–6 and 9.
