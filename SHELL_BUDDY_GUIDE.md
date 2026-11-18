# Shell Buddy: Fine-tune a Model and Deploy It, Step by Step

This guide records every lesson of the Shell Buddy project, from the first setup command to a public HTTPS server on AWS. Follow it top to bottom to rebuild the whole project from scratch.

**How to read it:**
- Each lesson starts with **why** we're doing it, then the steps.
- Steps are shown in their **final, working form**. Where something broke along the way, a **"⚠ Problem we hit"** box explains what happened and how it was fixed, so you'll recognise it if it happens again.
- `like this` = something you type or a file name. **Terminal L** = your laptop, **Terminal S** = the AWS server.

---

## The big picture

**One sentence:** we took a general AI model, taught it one job (turning English into Linux commands), and set it up as a tool that you, and later other people, can use.

**Analogy: training a new employee**

| Step | In the analogy | Why it was needed |
|---|---|---|
| 1. Download Qwen (Lessons 1–2) | Hire someone who already speaks English | Training a model from scratch costs millions. We start from one that already understands language. |
| 2. Prepare data (Lesson 3) | Write a training manual with 10,000 "when asked X, answer Y" examples | The model learns only from examples. Bad examples teach bad habits. |
| 3. Fine-tune (Lesson 4) | The employee studies the manual | The actual teaching. Runs on Colab because it needs a strong GPU. |
| 4. Evaluate (Lesson 5) | Test the employee with new questions | Check whether training helped. (Right tool: 62% → 92%.) |
| 5. Merge (Lesson 6) | Study notes become permanent knowledge | Training made a small add-on file; merging bakes it into one finished model. |
| 6. Build `howto` (Lesson 6) | Give the employee a desk where you can ask questions | Type a request in the terminal, get a command back. |
| 7. Server (Lesson 7) | The employee stays at the desk all day | Loading the model took ~8 s every time. A server loads it once. |
| 8. Docker (Lesson 8) | Pack the whole desk into a box you can move | Runs on any computer without setting everything up again. |
| 9. Quantization (Lesson 9) | Shrink the manual to pocket size | 531 MB instead of ~1 GB, much faster, runs on a small GPU. |
| 10. GitHub | Keep a copy of the recipe in a safe | Backup, history, easy download anywhere. |
| 11. AWS (Lessons A–F) | Open an office in the cloud | Shell Buddy runs on a rented server that others can reach safely. |

**Final architecture**

```
Your laptop                                   AWS server (Sydney)
howto-cloud "request"  ──HTTPS + API key──►   Caddy (443) → gateway.py (8000) → Ollama (11434) → shell-buddy
   │                                                          checks key, 20 req/min
   └─ shows command → danger warnings → you approve → runs on YOUR laptop
```

---

## Your setup at a glance

| Item | Value |
|---|---|
| Laptop | 8 CPU cores, 15 GB RAM, NVIDIA MX230 (2 GB, too old for PyTorch training, but Ollama can use it) |
| Project folder | `~/projects/model_build` |
| Python environment | `.mb` (Python 3.10.12). Note: Python 3.10 is end-of-life since Oct 2026; move to 3.12+ for anything serious. |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `AnishJoshi/nl2bash-custom` on Hugging Face (24,573 rows → 10,779 after cleaning) |
| Library versions | torch 2.14.1+cpu, transformers 5.20.2, fastapi 0.162.0, uvicorn 0.54.0, TRL 1.16 (Colab), Ollama 0.40.14 |
| Ports | 8765 = local FastAPI/Docker, 11434 = Ollama, 11435 = SSH tunnel on laptop, 8000 = gateway on the server |
| AWS | New sign-up flow (Builder ID, Free plan), region **Asia Pacific (Sydney)** |
| Public address | `https://<ELASTIC_IP>.sslip.io` (yours: `16.178.171.116.sslip.io`) |

**Final scores (100 unseen test examples)**

| Model | clean_format | valid_syntax | same_program | exact_match |
|---|---|---|---|---|
| Base Qwen 0.5B (not trained) | 98% | 99% | 62% | 5% |
| Qwen 7B (14× bigger, not trained) | 99% | 100% | 63% | 9% |
| Shell Buddy v2 (16-bit, PyTorch) | 100% | 100% | 92% | 19% |
| **Shell Buddy v2 (8-bit, Ollama)** | **100%** | **99%** | **91%** | **18%** |

---

## Contents

**Part 1: Build the model**
- [Lesson 1: Set up your machine](#lesson-1-set-up-your-machine)
- [Lesson 2: Talk to the base model](#lesson-2-talk-to-the-base-model)
- [Lesson 3: Prepare the dataset](#lesson-3-prepare-the-dataset)
- [Lesson 4: Fine-tune on Google Colab](#lesson-4-fine-tune-on-google-colab)
- [Lesson 5: Measure whether fine-tuning worked](#lesson-5-measure-whether-fine-tuning-worked)
- [Lesson 6: Merge the model and build `howto`](#lesson-6-merge-the-model-and-build-howto)

**Part 2: Serve it**
- [Lesson 7: Deploy as an API server](#lesson-7-deploy-as-an-api-server)
- [Lesson 8: Package with Docker](#lesson-8-package-with-docker)
- [Lesson 9: Quantization with Ollama](#lesson-9-quantization-with-ollama)

**Part 3: Save it**
- [GitHub lesson: upload your code](#github-lesson-upload-your-code)

**Part 4: Put it in the cloud**
- [AWS Lesson A: Account and safety](#aws-lesson-a-account-and-safety)
- [AWS Lesson B: Rent a cloud computer](#aws-lesson-b-rent-a-cloud-computer)
- [AWS Lesson C: Put Shell Buddy on the server](#aws-lesson-c-put-shell-buddy-on-the-server)
- [AWS Lesson D: Use it privately from your laptop](#aws-lesson-d-use-it-privately-from-your-laptop)
- [AWS Lesson E: Let other people use it safely](#aws-lesson-e-let-other-people-use-it-safely)
- [AWS Lesson F: Switching off and cleaning up](#aws-lesson-f-switching-off-and-cleaning-up)

**Appendix**
- [Daily cheat sheet](#daily-cheat-sheet)
- [Every problem we hit, in one table](#every-problem-we-hit-in-one-table)
- [Glossary](#glossary)
- [Safety rules](#safety-rules)
- [What's next](#whats-next)

---

# Part 1: Build the model

## Lesson 1: Set up your machine

### Key ideas

| Term | What it means |
|---|---|
| **Model** | A big file of numbers ("weights") that takes in text and predicts the next word, over and over. |
| **Pretrained model** | A model someone already trained on huge amounts of text. We start from one, because training from scratch costs millions. |
| **Fine-tuning** | Training a pretrained model a bit more on *your* examples so it learns a specific task or format. |
| **LoRA** | A cheap way to fine-tune: freeze the original weights and train a small "adapter" (a few MB) on top. |
| **Hugging Face** | The "GitHub of models". We download models and datasets from there. |
| **Deployment** | Running the model as a service (an API) so other programs can send it requests. |

### Step 1.1: Check your machine

```bash
nvidia-smi                 # NVIDIA GPU? How much memory?
free -h                    # How much RAM?
nproc                      # How many CPU cores?
df -h ~                    # How much free disk? (you need 10–15 GB)
python3 --version          # 3.10 or newer
```

**What the results meant for this project:**

| Part | What we had | Meaning |
|---|---|---|
| GPU | GeForce MX230, 2 GB | Too small and too old for training. Current PyTorch doesn't support it. **Not used for training.** |
| RAM | 15 GB | Fine for running a small model on the CPU |
| CPU | 8 cores | Fine for running and serving |
| Disk | 5.9 GB free (98% full) | **The real problem.** Needed ~15 GB free. |

**The plan this led to:** run, test and serve on the laptop (CPU), and do only the training step on **Google Colab's free GPU**. Real projects often do the same: train on a rented GPU, serve somewhere cheaper.

### Step 1.2: Free up disk space

Find what's using space:
```bash
du -h --max-depth=1 ~ 2>/dev/null | sort -h | tail -15      # biggest folders in home
du -sh ~/.cache/* 2>/dev/null | sort -h | tail -10          # caches (often huge)
```

Usually safe to clean:
```bash
pip cache purge                          # old downloaded Python packages
sudo apt clean                           # old downloaded system packages
sudo journalctl --vacuum-size=200M       # old system logs
```

Also check `docker system df`, `~/Downloads`, and old `node_modules` / `.venv` folders. Run `df -h ~` again. Don't continue until you have at least 10 GB free. (We went from 5.9 GB to 30 GB free.)

### Step 1.3: Create an isolated Python environment

A **virtual environment** keeps this project's libraries separate from the rest of your system.

```bash
cd ~/projects/model_build
python3 -m venv .mb
source .mb/bin/activate
pip install --upgrade pip
```

Your prompt now starts with `(.mb)`. Run `source .mb/bin/activate` again in every new terminal. (If `venv` errors out: `sudo apt install python3-venv`.)

### Step 1.4: Install the libraries (CPU-only PyTorch)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate peft trl
```

The CPU-only PyTorch is ~200 MB instead of several GB of GPU libraries you can't use.

| Library | What it does |
|---|---|
| `torch` | PyTorch, the math engine that runs the model |
| `transformers` | Loads models and tokenizers from Hugging Face |
| `datasets` | Loads and processes training data |
| `accelerate` | Places the model on GPU or CPU automatically |
| `peft` | Implements LoRA |
| `trl` | Provides `SFTTrainer`, the fine-tuning loop |

### Step 1.5: Check it worked

```bash
python --version
python -c "import torch; print('torch', torch.__version__); print('GPU available:', torch.cuda.is_available())"
```

`GPU available: False` is what we want here.

---

## Lesson 2: Talk to the base model

### Why

Before changing a model, see how it behaves as-is. This is your "before" baseline.

### Step 2.1: Create `run_base.py`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. The tokenizer converts text <-> numbers (tokens) the model understands
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Download (first time only, ~1 GB) and load the model weights
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float32")


# 3. Chat models expect messages in a specific format with roles
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain what fine-tuning is in one sentence."},
]

# 4. Turn the messages into the exact prompt string this model was trained on
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== PROMPT THE MODEL ACTUALLY SEES ===\n", prompt)

# 5. Tokenize, generate, decode
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
new_tokens = output[0][inputs["input_ids"].shape[1]:]   # skip the prompt, keep only the answer
print("=== ANSWER ===\n", tokenizer.decode(new_tokens, skip_special_tokens=True))

print("inputs keys:", inputs.keys())
print("prompt shape:", inputs["input_ids"].shape)
print("output shape:", output.shape)
print("new tokens count:", len(new_tokens))
print("FULL decode (prompt + answer):\n", tokenizer.decode(output[0]))
```

```bash
python run_base.py
```

The first run downloads ~1 GB into `~/.cache/huggingface/`. On a CPU an answer takes ~5–20 s.

**Why `float32` and no `device_map`?** The model is stored in `bfloat16`, which many laptop CPUs handle slowly. `float32` uses ~2 GB RAM and runs reliably. With only a CPU, there's no device to choose.

### Understanding the code

**Is Qwen a tokenizer or a model?** Qwen is an **LLM** (a family of models by Alibaba). Every model ships with its **own matching tokenizer** in the same Hugging Face repo, which is why `model_id` is used twice.

| File in the repo | Belongs to | What it is |
|---|---|---|
| `model.safetensors` | Model | The weights, ~1 GB of learned numbers |
| `config.json` | Model | The architecture (layers, sizes) |
| `tokenizer.json`, `vocab.json`, `merges.txt` | Tokenizer | Vocabulary of ~150,000 text pieces, each with an ID |
| `tokenizer_config.json` | Tokenizer | Settings, including the **chat template** |

```
"Hello there" → tokenizer → [9707, 1052] → model → [13, 2585, ...] → tokenizer → "! How can I help?"
```

The tokenizer is a fixed lookup table. The model is the "brain". They must match: Qwen's model with Llama's tokenizer gives garbage. Fine-tuning changes only the **model**; the tokenizer stays the same and is saved next to it.

Try it in a Python shell:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ids = tok.encode("Fine-tuning is fun")
print(ids)                                   # the numbers
print(tok.convert_ids_to_tokens(ids))        # the text pieces
print(tok.decode(ids))                       # back to text
```

**`apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`** turns your list of messages into the exact text format Qwen was trained on:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain what fine-tuning is in one sentence.<|im_end|>
<|im_start|>assistant

```

- `<|im_start|>` / `<|im_end|>` are special tokens marking each message.
- `tokenize=False` returns a readable string (instead of token IDs).
- `add_generation_prompt=True` adds the empty `<|im_start|>assistant` line so the model's next step is to write the reply.

| When | `add_generation_prompt` | Why |
|---|---|---|
| Inference (asking) | `True` | You want the model to write the reply |
| Training | `False` | The example already contains the answer |

**The three generation lines:**

```python
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
```
Splits the prompt into token IDs as PyTorch **tensors** (`"pt"`), on the same device as the model. `inputs` has `input_ids` and `attention_mask` (1 = "pay attention to this token"). Shape is `[batch_size, number_of_tokens]`, e.g. `[1, 30]`.

```python
output = model.generate(**inputs, max_new_tokens=100)
```
A loop: read all tokens → predict one next token → append → repeat, until `<|im_end|>` or 100 new tokens. `output` contains **prompt + answer**.

```python
new_tokens = output[0][inputs["input_ids"].shape[1]:]
```
Cuts off the first N tokens (the prompt) and keeps only the answer. `decode(..., skip_special_tokens=True)` turns it back into text.

Experiment: change `max_new_tokens=100` to `10` and see the answer cut off.

---

## Lesson 3: Prepare the dataset

### Why

The dataset matters more than anything else. A small model trained on clean, consistent examples beats one trained on lots of messy ones. About 80% of a real fine-tuning project is data work.

### Step 3.1: Choose the task

Fine-tuning works well for a **narrow, repeated task with a fixed output shape** (JSON, commands, SQL). It works poorly for teaching new facts (that's RAG) or making a small model smarter in general.

**Chosen project: Shell Buddy.** English request → one Linux bash command.
- A public dataset exists (no need to write examples by hand).
- Outputs are short (5–20 tokens): fast training, fast answers on CPU.
- Measurable: does the command match? Does it use the right program?
- Useful result: a `howto` terminal command.
- **Safety rule from day one:** the tool never runs a command automatically.

### Step 3.2: The data format

Training data is **JSONL** (one JSON object per line) in the `messages` format:

```json
{"messages": [
  {"role": "system", "content": "You are Shell Buddy. Convert the user's request into a single Linux bash command. Reply with only the command, no explanation and no markdown."},
  {"role": "user", "content": "find files under /etc larger than 100k"},
  {"role": "assistant", "content": "find /etc -size +100k"}
]}
```

The model sees the system and user messages and learns to produce that exact assistant reply.

### Step 3.3: Look at the data first

Dataset: [`AnishJoshi/nl2bash-custom`](https://huggingface.co/datasets/AnishJoshi/nl2bash-custom), ~24,600 English → bash pairs (from NL2Bash and NLC2CMD). Columns: `nl_command`, `bash_code`. It's messy, which is good practice. (No license stated: fine for learning; check before commercial use.)

**Never train on data you haven't read.** Create `explore_data.py`:

```python
import random
from collections import Counter
from datasets import load_dataset

ds = load_dataset("AnishJoshi/nl2bash-custom")
print(ds)   # shows the splits and row counts

train = ds["train"]
random.seed(0)
print("\n=== 15 random examples ===")
for i in random.sample(range(len(train)), 15):
    row = train[i]
    print(f"\nNL:   {row['nl_command']}\nBASH: {row['bash_code']}")

cmds = train["bash_code"]
first_words = Counter(c.split()[0] for c in cmds if c.split())
print("\n=== 20 most common programs ===")
print(first_words.most_common(20))

lengths = [len(c) for c in cmds]
print(f"\nCommand length (chars): min {min(lengths)}, max {max(lengths)}, avg {sum(lengths) / len(lengths):.0f}")
print("Multi-line commands:", sum("\n" in c for c in cmds))
print("Duplicate commands:", len(cmds) - len(set(cmds)))
```

```bash
cd ~/projects/model_build
source .mb/bin/activate
python explore_data.py
```

Run it a few times with a different `random.seed(...)` and write down problems you notice.

### Step 3.4: Cleaning rules

| Problem | Example | How we handle it |
|---|---|---|
| Typos in program name | `yargs -l 203` | Keep only if the program **exists** (`shutil.which`) |
| Broken syntax | `find project / src / -name "* .o"` | `bash -n` (checks syntax, **never runs anything**) |
| Multi-line scripts | `for f in ...; do\n...` | Drop |
| Very long commands | 300-character pipelines | Drop anything over 150 chars |
| Duplicates | Same request twice | Keep the first valid copy |
| Leakage between splits | Same pair in train and test | **Merge all splits, dedupe, make our own split** |
| Unquoted wildcards | `-name *.xml` | **Fix**: becomes `-name '*.xml'` |
| `-exec` without `\;` or `+` | `-exec touch -c {}` | Drop (`bash -n` can't see this; it's `find`'s own rule) |
| Truncated commands | ends with `\` | Drop |
| macOS `sed -i ''` | `sed -i '' 's/a/b/' f` | **Fix**: becomes `sed -i 's/a/b/' f` |
| Curly quotes | `‘*.mpg’`, `“man*”` | **Fix**: convert to straight quotes |
| Description doesn't match command | "text files" → `-name ".txt"` | Can't detect automatically: estimate by reading a sample |

**Three ideas to remember:**
1. **Leakage.** If a test example also appears in training, a good score only means memorization.
2. **Filtering is a trade-off.** Some valid commands get dropped too. With 24k rows, removing too much beats keeping junk.
3. **Fix vs. drop.** Fix a row only when the correct meaning is obvious (adding quotes). Drop it when you can't know what was intended (truncated command).

### Step 3.5: The cleaning script (final version)

Create `clean_data.py`:

```python
import json
import random
import shutil
import subprocess
from collections import Counter
from datasets import load_dataset, concatenate_datasets
import os
import re

# The model learns to respond to THIS exact prompt; we'll reuse it in deployment
SYSTEM_PROMPT = (
    "You are Shell Buddy. Convert the user's request into a single Linux bash command. "
    "Reply with only the command, no explanation and no markdown."
)
MAX_CMD_CHARS = 150
TEST_SIZE = 300
# Shell builtins aren't files on disk, so shutil.which() can't find them
BASH_BUILTINS = {"cd", "export", "source", "alias", "unset", "read", "set", "pushd", "popd",
                 "history", "ulimit", "umask", "type", "declare", "shopt", "exit", "wait", "trap"}
# Blog copy-paste turns ' and " into curly quotes; bash treats those as literal characters
SMART_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def program_exists(cmd):
    words = cmd.split()
    if words[0] == "sudo" and len(words) > 1:
        words = words[1:]
    return words[0] in BASH_BUILTINS or shutil.which(words[0]) is not None


def valid_syntax(cmd):
    # bash -n = parse only, NEVER executes the command
    result = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, timeout=5)
    return result.returncode == 0

GLOB_ARG = re.compile(r"(-i?name|-i?path|-i?regex)\s+([^'\"\s\\]*[*?][^'\"\s]*)")


def quote_globs(cmd):
    # find -name *.txt  ->  find -name '*.txt'  (unquoted globs get expanded by the shell first)
    return GLOB_ARG.sub(r"\1 '\2'", cmd)


def missing_exec_terminator(cmd):
    # find's -exec must end with \; or + ; bash -n can't see this, it's find's own rule
    return re.search(r"-exec(dir)?\s", cmd) is not None and re.search(r"\\;|';'|\";\"|\}\s*\+", cmd) is None


def to_chat(row):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["request"]},
            {"role": "assistant", "content": row["command"]},
        ]
    }


ds = load_dataset("AnishJoshi/nl2bash-custom")
all_rows = concatenate_datasets([ds[split] for split in ds])   # merge, we re-split later
print("Loaded rows:", len(all_rows))

drops = Counter()
fixes = 0
seen_requests = set()
kept = []
for row in all_rows:
    request = " ".join(row["nl_command"].translate(SMART_QUOTES).split())
    cmd = row["bash_code"].strip().translate(SMART_QUOTES)

    fixed_cmd = quote_globs(cmd)
    if fixed_cmd != cmd:
        fixes += 1
        cmd = fixed_cmd

    # macOS-style "sed -i ''" fails on Linux; GNU sed just wants "sed -i"
    linux_cmd = re.sub(r"sed -i (''|\"\")\s+", "sed -i ", cmd)
    if linux_cmd != cmd:
        fixes += 1
        cmd = linux_cmd

    if not request or not cmd:
        drops["empty"] += 1
    elif "\n" in cmd:
        drops["multi-line"] += 1
    elif cmd.endswith("\\"):
        drops["truncated"] += 1
    elif len(cmd) > MAX_CMD_CHARS:
        drops["too long"] += 1
    elif missing_exec_terminator(cmd):
        drops["-exec without terminator"] += 1
    elif request.lower() in seen_requests:
        drops["duplicate request"] += 1
    elif not program_exists(cmd):
        drops["unknown program"] += 1
    elif not valid_syntax(cmd):
        drops["syntax error"] += 1
    else:
        seen_requests.add(request.lower())
        kept.append({"request": request, "command": cmd})

print("Fixed (quoted globs):", fixes)
print("Dropped:", dict(drops))
print("Kept:", len(kept))

random.seed(42)
random.shuffle(kept)
test_rows, train_rows = kept[:TEST_SIZE], kept[TEST_SIZE:]

os.makedirs("data", exist_ok=True)   # create data/ if missing; do nothing if it exists

for path, subset in (("data/train.jsonl", train_rows), ("data/test.jsonl", test_rows)):
    with open(path, "w") as f:
        for row in subset:
            f.write(json.dumps(to_chat(row)) + "\n")

# A random sample for YOU to review by eye
with open("data/review_sample.txt", "w") as f:
    for row in random.sample(train_rows, 50):
        f.write(f"NL:   {row['request']}\nBASH: {row['command']}\n\n")

print(f"Wrote train: {len(train_rows)}, test: {len(test_rows)}")
print("Top programs in train:", Counter(r["command"].split()[0] for r in train_rows).most_common(10))
```

```bash
cd ~/projects/model_build
python clean_data.py
```

It runs `bash -n` once per row, so it takes 1–3 minutes.

**How it works:**
- **`if/elif` chain:** each row is dropped for the *first* rule it fails; `drops` counts how many each rule removed.
- **`seen_requests` is only updated for kept rows:** if the first copy is broken and a later copy is fine, the good one is kept.
- **Cheap checks first:** the slow `bash -n` only runs on rows that passed everything else.
- **`random.seed(42)`** makes the split the same every run.

**Results (first run):** 24,573 loaded; **10,628 duplicate requests (43%!)**, 1,619 multi-line, 1,220 unknown program, 150 syntax error, 74 too long, 4 empty. **Final run: 10,779 kept** (10,479 train + 300 test), 459 fixed.

> **⚠ Problem we hit: `FileNotFoundError: 'data/train.jsonl'`**
> `open(path, "w")` creates files but not folders, and `data/` didn't exist. Fixed with `os.makedirs("data", exist_ok=True)` (already in the script above). Also remember: `"data/train.jsonl"` is relative to the folder your **terminal** is in (`pwd`), so always `cd ~/projects/model_build` first.

> **⚠ Problem we hit: curly quotes taught the model a bug**
> Some commands had blog-style curly quotes (`‘ ’ “ ”`). Bash treats them as ordinary characters, and our quote-fixing wrapped them in real quotes (`'‘*.mpg’'`). The trained model then produced broken commands like `find / -type d -name '“man*”'`. `bash -n` passed them because they're valid bash; they just don't work. The `SMART_QUOTES` line fixes it. Check with:
> ```bash
> grep -cE '\\u201[89cd]' data/train.jsonl     # must print 0
> ```
> Lesson: **garbage in, garbage out.** The model faithfully learns the data's mistakes.

### Step 3.6: Check quality by hand

Open `data/review_sample.txt` and read all 50 rows. For each, ask: **does the command do what the description says?**
- Under 5 wrong (<10%): good.
- 5–10 wrong: usable.
- More than 10: find the pattern and add a rule.

Our review found ~10% wrong, mostly mismatched descriptions that no code can catch. **When to stop cleaning:** each new rule fixes fewer rows than the last. A model trained on ~90% correct data learns the common pattern. Risky commands (like `| crontab`) are handled later by **guardrails in the tool**, not in the data.

### Step 3.7: Check token lengths

Create `inspect_data.py`:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ds = load_dataset("json", data_files={"train": "data/train.jsonl", "test": "data/test.jsonl"})
print(ds)

# What one training example looks like after the chat template
print(tokenizer.apply_chat_template(ds["train"][0]["messages"], tokenize=False))

# Token lengths: needed to pick max_length for training in Lesson 4
lengths = [len(tokenizer(tokenizer.apply_chat_template(ex["messages"], tokenize=False))["input_ids"]) for ex in ds["train"]]
print(f"Token lengths → max: {max(lengths)}, avg: {sum(lengths) / len(lengths):.0f}")
```

```bash
python inspect_data.py
```

Our result: **max 188 tokens, average 78** → we'll use `max_length=256` in training.

### Step 3.8: Record the "before" answers

In `run_base.py`, change the messages to the Shell Buddy prompt and try 3–4 requests:

```python
messages = [
    {"role": "system", "content": "You are Shell Buddy. Convert the user's request into a single Linux bash command. Reply with only the command, no explanation and no markdown."},
    {"role": "user", "content": "find all files bigger than 1GB in my home folder"},
]
```

Look for markdown fences, explanations and wrong flags.

---

## Lesson 4: Fine-tune on Google Colab

### What happens during training

```
for each batch of 16 examples:
    1. The model reads the prompt and predicts the answer tokens one by one
    2. Loss = how wrong those predictions were (lower is better)
    3. Work out which direction to nudge the weights to reduce the loss
    4. Nudge ONLY the LoRA adapter weights a little
repeat for every batch → 1 epoch. We do 2 epochs.
```

**Three ideas:**
1. **Train on the answer only.** The system prompt is identical in every row, so there's nothing to learn from it. We split each example into a `prompt` (system + user) and a `completion` (the command). TRL then computes the loss only on the completion.
2. **LoRA trains ~1.75% of the model.** The 494M original weights stay frozen; ~9M new weights are trained. Result: a ~35 MB adapter instead of a 1 GB model.
3. **Three data sets:**

| Set | Rows | Used for |
|---|---|---|
| train | ~10,177 | Training |
| validation | 300 (taken from train.jsonl) | Checking progress *during* training |
| test | 300 (test.jsonl) | **Never uploaded to Colab.** Final scoring in Lesson 5. |

### The settings (hyperparameters)

| Setting | Value | Why |
|---|---|---|
| `num_train_epochs` | 2 | Each example seen twice. More risks memorizing. |
| `per_device_train_batch_size` | 16 | Examples per step. Short examples, so 16 fits. |
| `learning_rate` | 2e-4 | Size of each nudge. LoRA uses higher rates than full fine-tuning. |
| `max_length` | 256 | Longest example is 188 tokens |
| `r` / `lora_alpha` | 16 / 32 | Adapter size and strength (standard starting values) |
| `fp16=True, bf16=False` | | Colab's T4 supports fp16, not bf16. TRL defaults to bf16, so this must be set. |

### Step 4.1: Set up Colab

1. [colab.research.google.com](https://colab.research.google.com) → **New notebook**
2. **Runtime → Change runtime type → T4 GPU → Save**
3. Rename the notebook to `shell_buddy_training`

Each block below is **one cell**. Paste, press **Shift+Enter**, wait for it to finish.

**Cell 1: check the GPU**
```python
!nvidia-smi
```
You should see a **Tesla T4** with ~15 GB.

**Cell 2: install libraries**
```python
!pip install -q -U trl peft datasets
```

**Cell 2b: remove torchao, then restart**
```python
!pip uninstall -y torchao
```
Then **Runtime → Restart session**. Don't re-run Cell 2 afterwards.

> **⚠ Problem we hit: `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`**
> Colab pre-installs torchao 0.10. The newly upgraded PEFT checks its version and refuses to start. We don't use torchao, so removing it is safer than upgrading it (upgrading could break PyTorch). Restarting the session makes Python forget the old check. A restart **keeps files** but **clears variables**, so re-run from Cell 4.
> How to read a traceback: start at the **bottom line** (the actual error), then look at which file raised it.

**Cell 3: upload the training data**
```python
from google.colab import files
uploaded = files.upload()
```
Choose **only** `~/projects/model_build/data/train.jsonl`. Colab files are temporary.

**Cell 4: convert to prompt/completion and split off validation**
```python
from datasets import load_dataset

ds = load_dataset("json", data_files="train.jsonl", split="train")

# messages → prompt (system + user) and completion (assistant)
def to_prompt_completion(ex):
    return {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]}

ds = ds.map(to_prompt_completion, remove_columns=["messages"])

split = ds.train_test_split(test_size=300, seed=42)
train_ds, val_ds = split["train"], split["test"]
print(train_ds)
print(val_ds)
print(train_ds[0])
```
`[:-1]` = every message except the last; `[-1:]` = only the last.

**Cell 5: configure LoRA and the trainer**
```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",   # attach adapters to every linear layer
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="shell-buddy-lora",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",    # learning rate slowly decreases toward the end
    warmup_steps=30,               # ...and starts small for the first 30 steps
    max_length=256,
    fp16=True,
    bf16=False,
    gradient_checkpointing=False,  # small model, plenty of memory → faster without it
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,                # check validation loss every 100 steps
    save_strategy="no",            # we save manually at the end
    report_to="none",
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
)
trainer.model.print_trainable_parameters()
```
Expect something like `trainable params: ~8.8M || all params: ~503M || trainable%: ~1.75`.

**Cell 6: train**
```python
trainer.train()
```
~1,274 steps, about **11 minutes** on a T4. Keep the tab open.

**Cell 7: quick test inside Colab**
```python
SYSTEM_PROMPT = (
    "You are Shell Buddy. Convert the user's request into a single Linux bash command. "
    "Reply with only the command, no explanation and no markdown."
)
model = trainer.model
tokenizer = trainer.processing_class
model.eval()

def ask(request):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

for request in [
    "find all files bigger than 1GB in my home folder",
    "show which folders use the most disk space",
    "count lines in all python files in this directory",
    "delete empty directories under /tmp",
    "show the 10 most recently modified files",
]:
    print(f"{request}\n  → {ask(request)}\n")
```
`do_sample=False` = always pick the most likely token, so the same request gives the same answer.

**Cell 8: save and download the adapter** (use a new name for each version)
```python
trainer.save_model("shell-buddy-lora-v2")
!zip -r shell-buddy-lora-v2.zip shell-buddy-lora-v2
files.download("shell-buddy-lora-v2.zip")
```

On your laptop:
```bash
cd ~/projects/model_build
mv ~/Downloads/shell-buddy-lora-v2.zip .
unzip shell-buddy-lora-v2.zip
ls shell-buddy-lora-v2      # adapter_config.json, adapter_model.safetensors (~35 MB), tokenizer files
```

Also save the notebook: **File → Download → Download .ipynb** into the project folder as `shell_buddy_training.ipynb`.

### How to read the training log

| Column | What you want to see |
|---|---|
| Training Loss | Drops quickly, then levels off |
| Validation Loss | Also goes down. **If it starts rising while training loss keeps falling, the model is overfitting (memorizing).** |
| Mean Token Accuracy | Goes up |

**What our run showed (v1):**
- Validation loss went 1.08 → 0.83 and **never went up**: no overfitting. It flattened near the end, so 2 epochs was right.
- **Training loss dropped sharply at step ~637** (0.90 → 0.64). That's where epoch 2 starts and the model sees the same examples again. Validation loss barely moved, because those examples are new to it. **Training loss shows memorization; validation loss shows real learning.** Watch both.
- Validation loss 0.83 ≈ choosing between ~2.3 equally likely tokens (e^0.83). Loss can't reach 0: many requests have several correct commands, and ~10% of labels are wrong.
- 80% token accuracy ≠ 80% correct commands. During training the model is given the correct previous tokens; in real use one wrong token can derail the rest.

**v2 (after the curly-quote fix):** final validation loss **0.742** (v1: 0.834), token accuracy 80.3%, ~10.8 minutes.

---

## Lesson 5: Measure whether fine-tuning worked

### Why

A few good answers can be luck. We score **both** the base and fine-tuned model on the 300 test examples the model never saw.

### How a LoRA model loads

The adapter **attaches to the base model**:
```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "shell-buddy-lora-v2")
```
Trick: `with model.disable_adapter():` temporarily gives you the original base model, so one script can score both.

### What we measure

| Metric | Question |
|---|---|
| `clean_format` | Is the reply **only** the command (no fences, no explanation)? |
| `valid_syntax` | Does it pass `bash -n`? |
| `same_program` | Right tool? (e.g. `find` when `find` was expected) |
| `exact_match` | Same as expected, ignoring spacing and quote style? |

Format and content are scored separately: the script pulls the command out of any ```` ``` ```` fences first, so the base model gets a fair content score.

**Why `exact_match` is low:** a request can have several correct commands (`find . -name '*.txt'` vs `find -name "*.txt"`), and ~10% of test labels are wrong. Compare base vs. tuned, and read mismatches by hand.

### Step 5.1: The evaluation script

Create `evaluate.py`:

```python
import json
import re
import subprocess
import sys
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100   # how many test rows to score
ADAPTER_DIR = sys.argv[2] if len(sys.argv) > 2 else "shell-buddy-lora-v2"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()


def generate(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():   # no training here, so skip gradient bookkeeping (faster, less memory)
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract_command(text):
    # If the model used a ```bash ... ``` block, take what's inside; then keep the first line
    match = re.search(r"```(?:bash|sh)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1)
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def clean_format(text):
    return "```" not in text and "\n" not in text


def normalize(cmd):
    return " ".join(cmd.replace('"', "'").split())


def valid_syntax(cmd):
    return subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, timeout=5).returncode == 0


with open("data/test.jsonl") as f:
    rows = [json.loads(line) for line in f][:N]

scores = {"base": Counter(), "tuned": Counter()}
results = []
for i, row in enumerate(rows, start=1):
    prompt_messages = row["messages"][:-1]
    expected = row["messages"][-1]["content"]

    with model.disable_adapter():
        base_out = generate(prompt_messages)
    tuned_out = generate(prompt_messages)

    for name, out in (("base", base_out), ("tuned", tuned_out)):
        cmd = extract_command(out)
        s = scores[name]
        s["clean_format"] += clean_format(out)
        s["valid_syntax"] += bool(cmd) and valid_syntax(cmd)
        s["same_program"] += bool(cmd) and cmd.split()[0] == expected.split()[0]
        s["exact_match"] += normalize(cmd) == normalize(expected)

    results.append({"request": prompt_messages[-1]["content"], "expected": expected,
                    "base": base_out, "tuned": tuned_out})
    print(f"Scored {i}/{len(rows)}", end="\r", flush=True)

print(f"\n\n{'metric':<15}{'base':>8}{'tuned':>8}")
for metric in ("clean_format", "valid_syntax", "same_program", "exact_match"):
    print(f"{metric:<15}{scores['base'][metric] / len(rows):>8.0%}{scores['tuned'][metric] / len(rows):>8.0%}")

with open(f"data/eval_results_{ADAPTER_DIR}.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("\n=== 5 cases where the tuned model differs from expected ===")
misses = [r for r in results if normalize(extract_command(r["tuned"])) != normalize(r["expected"])]
for r in misses[:5]:
    print(f"\nREQUEST:  {r['request']}\nEXPECTED: {r['expected']}\nTUNED:    {r['tuned']}")
```

**Parts worth understanding:**
- `with torch.no_grad()`: skip training bookkeeping when only generating.
- `s["clean_format"] += clean_format(out)`: `True` counts as 1, so adding booleans counts passes.
- `bool(cmd) and ...`: skips the check when the command is empty, so `split()[0]` can't crash.
- `end="\r"`: the progress counter updates in place.
- The `N` line must come before `ADAPTER_DIR`, since both read `sys.argv`.

### Step 5.2: Run it

```bash
cd ~/projects/model_build
source .mb/bin/activate
python evaluate.py 100 shell-buddy-lora-v2     # ~5–10 minutes on CPU
python evaluate.py 100 shell-buddy-lora        # optional: v1 on the same test set
```

**Results on the current test set:**

| Metric | Base | Tuned v2 |
|---|---|---|
| clean_format | 98% | **100%** |
| valid_syntax | 99% | **100%** |
| same_program | 62% | **92%** |
| exact_match | 5% | **19%** |

> **⚠ Problem we hit: scores that can't be compared**
> After re-running `clean_data.py`, 2 extra rows were kept, the shuffle changed, and **the 300 test rows were different**. The base column changed (53% → 62% same_program) even though the base model hadn't. Lesson: only compare models scored on **the same test set**. If the base column matches between two runs, the test set is the same.

### Step 5.3: Read the mismatches yourself

For each mismatch, decide:
1. **Model wrong**: a real error.
2. **Both correct**: a different command that also works.
3. **Label wrong**: the model is right and the expected answer is bad data.

In our v2 sample, only 1 of 5 was a real error (`hostname -I` prints the IP, not the name). The others were equally correct, better, or had broken labels. So real accuracy is much higher than `exact_match` suggests.

### The improvement loop

This is how models get better: **find the problem in evaluation → fix the data → retrain → evaluate again.** v1 → v2 was exactly this (the curly-quote bug).

Known weakness found here: **63% of training data is `find`**, so the model is weak at everyday commands (`pwd`, `df -h`, `du`, `sort -r`). This is **data imbalance**. Fix later as v3.

---

## Lesson 6: Merge the model and build `howto`

### Why merge?

| | Base + adapter | Merged |
|---|---|---|
| Libraries to run it | transformers **and** peft | transformers only |
| Speed | Slightly slower | Same as the original |
| Works with Ollama, vLLM, etc. | Needs extra setup | Directly |
| Files | Two pieces | One folder |

The adapter is for training and experiments. The merged model is what you **ship**.

### Step 6.1: Merge

Check you have ~2 GB free (`df -h ~`). Create `merge_model.py`:

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ADAPTER_DIR = "shell-buddy-lora-v2"
OUTPUT_DIR = "shell-buddy-merged"

# Reads adapter_config.json → finds the base model → loads base + adapter together
model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER_DIR, dtype=torch.float32)

# Add the LoRA changes into the base weights, then drop the adapter layers
merged = model.merge_and_unload()

# Save in bfloat16: half the size of float32 (~1 GB instead of ~2 GB)
merged = merged.to(torch.bfloat16)
merged.save_pretrained(OUTPUT_DIR)
AutoTokenizer.from_pretrained(ADAPTER_DIR).save_pretrained(OUTPUT_DIR)
print("Saved merged model to", OUTPUT_DIR)
```

```bash
cd ~/projects/model_build
source .mb/bin/activate
python merge_model.py
ls -lh shell-buddy-merged     # model.safetensors (~1 GB), config.json, tokenizer files
```

Merge in float32 for accuracy, save in bfloat16 for size, load as float32 on the CPU for speed.

### Step 6.2: Build `howto.py` (first version)

The first version loaded the model directly. The **final** `howto.py` (with server, Ollama and cloud support) is in [Lesson 9](#step-98-switch-howto-to-ollama) and [AWS Lesson E](#step-e11-teach-howto-to-use-the-public-server). The ideas introduced here stay the same:

- **The default answer is always "no".** Enter, anything other than `y`, or Ctrl+C all cancel. **Risky commands require typing the full word `yes`.**
- **Danger warnings are simple fixed rules around an unpredictable model.** The model will sometimes be wrong (we saw `sort -n` showing the *oldest* files instead of the newest), so predictable checks wrap it.
- **`Path(__file__).resolve().parent`** finds the folder `howto.py` lives in, so it works from any directory.
- **`if __name__ == "__main__":`** runs `main()` only when you execute the file, not when another script imports it.

### Step 6.3: Guardrails we added after testing

Testing found three real problems:

| Test | Model answer | Problem | Guardrail |
|---|---|---|---|
| "what's the capital of France" | A URL | A fine-tuned model only knows one task; it can't refuse. The `&` in the URL even split it into two commands. | **Check the first word is a real program** (`shutil.which`). If not, refuse. |
| "show the current directory" | `cd -L .` | `subprocess.run` starts a **child process**. A child can never change your terminal's directory or variables. | **`cd`, `export`, `source`, `alias`, … → tell the user to copy and run it themselves.** |
| (any) | — | `shell=True` uses `/bin/sh` (dash on Ubuntu), but our data is bash | **`executable="/bin/bash"`** |

Also: **type straight quotes** (`"`), not curly ones (`“ ”`). Small input changes can change the output.

### Step 6.4: Make `howto` a terminal command

```bash
echo "alias howto='$HOME/projects/model_build/.mb/bin/python $HOME/projects/model_build/howto.py'" >> ~/.bashrc
source ~/.bashrc
type howto        # should print: howto is aliased to ...
```

`>>` **appends**. A single `>` would **overwrite** your whole `.bashrc`. Run the `echo` only once.

> **⚠ Problem we hit: `bin/env: No such file or directory` in every new terminal, and a duplicate alias**
> Line 123 of `~/.bashrc` (and line 29 of `~/.profile`) sourced a file inside VS Code's snap folder (`~/snap/code/264/...`) that no longer existed after VS Code updated. The `echo >>` had also been run twice. Fix: `nano ~/.bashrc`, put `#` in front of the broken `. "$HOME/snap/code/264/..."` line, delete the duplicate alias, do the same in `~/.profile`, then `source ~/.bashrc`. (Commenting out instead of deleting lets you undo it.)

### Step 6.5: Try it

```bash
howto "find all files bigger than 1GB in my home folder"
howto "count lines in all python files in this directory"
howto "delete empty directories under /tmp"          # should show a ⚠ warning
howto "what's the capital of France"                 # should refuse
howto "go to my Downloads folder"                    # should say: copy & run it yourself
time howto "list all hidden files"
```

**Always quote your request**: otherwise `?`, `*` and `>` are handled by the shell first. `time` showed ~8.8 s of CPU (`user`), almost all of it loading PyTorch and the 1 GB model **on every run**. Lesson 7 fixes that.

---

# Part 2: Serve it

## Lesson 7: Deploy as an API server

### Why a server?

Every `howto` call used to do this:
```
start Python → import PyTorch (~2s) → load 1 GB model (~5s) → generate (~1s) → exit and discard everything
```
A **server** loads the model **once** and keeps it in memory. Any program then sends it a request over **HTTP** and gets an answer:
```
                    ┌──────── server.py (runs all the time) ────────┐
howto  ──HTTP──▶    │  model already in memory → generate (~1s)    │
curl   ──HTTP──▶    │                                               │
a web app ──HTTP──▶ └───────────────────────────────────────────────┘
```
That's what "deploying a model" means: the model runs as a service behind an API.

### Where should each check run?

- **On the client (`howto`):** checks that depend on the user's machine. "Is this program installed?" and "Does this change the current shell?" only make sense where the command will run.
- **Anywhere:** danger warnings only look at the command text, so the server can return them.

### Step 7.1: Install the web framework

```bash
cd ~/projects/model_build
source .mb/bin/activate
pip install fastapi uvicorn
```
- **FastAPI** turns Python functions into API endpoints, validates input, and generates docs.
- **Uvicorn** is the web server that runs a FastAPI app.

### Step 7.2: `howto.py` as both client and library

Three changes: PyTorch imports move **inside** `load()`/`suggest()` (the server path never loads PyTorch); new `suggest_via_api()` tries the server first and falls back to the local model; guardrails move into `check_command()`.

```python
"""howto: turn an English request into a bash command using the fine-tuned Shell Buddy model."""
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Path relative to THIS file, so howto works from any folder
MODEL_DIR = Path(__file__).resolve().parent / "shell-buddy-merged"
API_URL = os.environ.get("SHELL_BUDDY_URL", "http://127.0.0.1:8765")
SYSTEM_PROMPT = (
    "You are Shell Buddy. Convert the user's request into a single Linux bash command. "
    "Reply with only the command, no explanation and no markdown."
)
DANGER_PATTERNS = [
    (r"\b(rm|rmdir|unlink|shred)\b|-delete\b", "deletes files"),
    (r"\bdd\b", "writes raw data to a disk"),
    (r"\bmkfs", "formats a disk"),
    (r"\bch(mod|own)\s+-R|chmod\s+777", "changes permissions/ownership on many files"),
    (r"\|\s*crontab\b", "REPLACES your entire crontab"),
    (r"\bkill(all)?\b|\bpkill\b", "stops running processes"),
    (r"\bsudo\b", "runs as root (administrator)"),
    (r"\b(shutdown|reboot|poweroff)\b", "shuts down or restarts the machine"),
    (r"\bmv\b|sed\s+-i", "moves or modifies files"),
    (r">\s*/dev/sd", "overwrites a disk device"),
]
# These change the CURRENT shell; a child process can't do that for you
SHELL_STATE_COMMANDS = {"cd", "export", "source", ".", "alias", "unset", "pushd", "popd"}
BASH_BUILTINS = {"read", "set", "history", "ulimit", "umask", "type", "declare", "shopt", "trap", "wait"}
BOLD, YELLOW, RESET = "\033[1m", "\033[33m", "\033[0m"


def load():
    # Heavy imports live here, so the API path never pays for loading PyTorch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging

    logging.set_verbosity_error()
    logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def suggest(tokenizer, model, request):
    import torch

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text.splitlines()[0].strip() if text else ""


def suggest_via_api(request):
    """Ask a running Shell Buddy server. Returns None if no server is reachable."""
    body = json.dumps({"request": request}).encode()
    req = urllib.request.Request(f"{API_URL}/suggest", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["command"]
    except (urllib.error.URLError, TimeoutError):
        return None


def danger_warnings(cmd):
    return [reason for pattern, reason in DANGER_PATTERNS if re.search(pattern, cmd)]


def first_program(cmd):
    words = cmd.split()
    if words and words[0] == "sudo" and len(words) > 1:
        words = words[1:]
    return words[0] if words else ""


def check_command(cmd):
    """Machine-specific checks. Returns (runnable, note); note explains why it isn't runnable."""
    program = first_program(cmd)
    if program in SHELL_STATE_COMMANDS:
        return False, "This changes your current shell (directory or variables). Copy it and run it yourself."
    if program not in BASH_BUILTINS and shutil.which(program) is None:
        return False, f"'{program}' isn't a program on this system, so this isn't a valid command. Not running it."
    return True, None


def main():
    if len(sys.argv) < 2:
        print('Usage: howto "describe what you want to do"')
        sys.exit(1)
    request = " ".join(sys.argv[1:])

    cmd = suggest_via_api(request)
    if cmd is None:
        print("(Shell Buddy server not running, loading the model locally...)")
        tokenizer, model = load()
        cmd = suggest(tokenizer, model, request)
    if not cmd:
        print("Shell Buddy had no suggestion.")
        sys.exit(1)

    print(f"\n  {BOLD}{cmd}{RESET}\n")
    runnable, note = check_command(cmd)
    if not runnable:
        print(f"  {note}")
        return

    warnings = danger_warnings(cmd)
    for w in warnings:
        print(f"  {YELLOW}⚠  This command {w}{RESET}")

    try:
        if warnings:
            run = input("\nType 'yes' to run it anyway, anything else to cancel: ").strip() == "yes"
        else:
            run = input("Run it? [y/N] ").strip().lower() == "y"
    except (KeyboardInterrupt, EOFError):
        run = False
        print()

    if run:
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    else:
        print("Not run.")


if __name__ == "__main__":
    main()
```

- **`os.environ.get("SHELL_BUDDY_URL", ...)`** reads the server address from an **environment variable**, with a default. Change the variable, not the code, to point at another server.
- **`urllib`** is Python's built-in HTTP client, so `howto` needs no extra installs.
- **`except URLError`** catches "connection refused" when the server is off, which triggers the local fallback.

### Step 7.3: The server

Create `server.py`:

```python
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
```

| Code | Concept |
|---|---|
| `@app.post("/suggest")` | An **endpoint**: URL path + HTTP method. `POST` sends data, `GET` reads. |
| `SuggestRequest(BaseModel)` | **Input validation.** Bad input (missing, empty, >300 chars) is rejected **before your code runs**. |
| `SuggestResponse` | The guaranteed **shape of the response** |
| `lifespan` | Runs at startup and shutdown: the model loads once |
| `/health` | Standard "am I up?" endpoint for monitoring and Docker |
| `def` (not `async def`) | FastAPI runs normal `def` endpoints in a thread pool, so the server still accepts connections while generating |
| `generate_lock` | Requests that arrive together wait their turn |

### Step 7.4: Run it

A **port** is a numbered door on your computer. Each server listens on its own. Port 8000 was already used by another app, so we use **8765**. Check it's free first:
```bash
ss -ltn | grep 8765        # no output = free
```

**Terminal 1** (stays open):
```bash
cd ~/projects/model_build
source .mb/bin/activate
uvicorn server:app --port 8765
```
Wait for `Application startup complete.` By default uvicorn listens only on **127.0.0.1** (this computer). **Don't use `--host 0.0.0.0`** without authentication: anyone on your Wi-Fi could use it.

> If the client's port doesn't match the server's, `howto` would hit the wrong app, get an error, and silently fall back to the slow local model. If `howto` is slow with the server running, check the ports.

### Step 7.5: Test it

1. Open **http://127.0.0.1:8765/docs**: FastAPI's free documentation page. Try **POST /suggest → Try it out → Execute**.
2. In a second terminal:
   ```bash
   curl -s http://127.0.0.1:8765/health
   time curl -s -X POST http://127.0.0.1:8765/suggest \
     -H "Content-Type: application/json" \
     -d '{"request": "find all png files in this folder"}'
   ```
3. Bad input (expect a **422** error that says exactly what's wrong):
   ```bash
   curl -s -X POST http://127.0.0.1:8765/suggest -H "Content-Type: application/json" -d '{"request": ""}'
   ```
4. `time howto "find all png files in this folder"` with the server running, then again after stopping it (Ctrl+C in Terminal 1).

**Our results:** `user` time **0.06 s** with the server vs **10.76 s** without: ~180× less work for the client. (`real` includes the time you spend typing Y. `user` > `real` in the local run because PyTorch uses several cores at once.)

---

## Lesson 8: Package with Docker

### Why Docker?

Your server works because of everything set up on **your** laptop. Docker packages all of it into one **image** that runs the same anywhere.

| Term | Meaning | Comparison |
|---|---|---|
| **Dockerfile** | Step-by-step build instructions | A recipe |
| **Image** | The finished, frozen package (OS files, Python, libraries, model, code) | The cooked dish |
| **Container** | A running instance of an image | A plate of that dish on the table |

### Step 8.1: Check what you have

```bash
docker --version
df -h ~                         # need at least 4 GB free; image is ~3.4 GB
.mb/bin/python --version        # the image should match: Python 3.10
```
No Docker? `sudo apt install docker.io && sudo usermod -aG docker $USER`, then log out and back in.

On Ubuntu, plain `python` only exists **inside** a venv. Outside it, the command is `python3`.

### Step 8.2: Pin exact versions

So the image doesn't install whatever is newest on build day (remember torchao):
```bash
source .mb/bin/activate
pip show torch transformers fastapi uvicorn | grep -E '^(Name|Version)'
```

`requirements-server.txt`:
```text
transformers==5.20.2
fastapi==0.162.0
uvicorn==0.54.0
```
torch gets its own install line. **peft isn't needed**: the server loads the merged model.

### Step 8.3: `.dockerignore`

Docker first copies the whole folder into its build area. Skip what doesn't belong in the image:
```text
.mb/
data/
shell-buddy-lora/
shell-buddy-lora-v2/
*.zip
__pycache__/
```

### Step 8.4: The Dockerfile (final)

Create `Dockerfile` (no extension):

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 0. The base image ships an old pip that mishandles PyTorch's package index
RUN pip install --no-cache-dir --upgrade pip

# 1. CPU-only PyTorch (the default wheel bundles ~3 GB of GPU libraries we can't use)
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Remaining dependencies, pinned
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# 3. A non-root user to run the app (created BEFORE copying, so it can own the files)
RUN useradd --create-home appuser

# 4. The model (rarely changes), then the code (changes often)
COPY --chown=appuser:appuser shell-buddy-merged/ ./shell-buddy-merged/
COPY --chown=appuser:appuser howto.py server.py ./

USER appuser

# 5. Docker checks /health to know if the app is actually working
HEALTHCHECK --interval=30s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health')"

EXPOSE 8765
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]
```

- **Layers and caching:** each instruction is a layer, and unchanged layers are reused. Order from **least to most often changed**, so a code edit doesn't reinstall PyTorch.
- **`torch==2.14.1`** without `+cpu`: the `--index-url .../whl/cpu` selects the CPU build.
- **`--no-cache-dir`:** don't keep downloaded packages inside the image.
- **`python:3.10-slim`:** ~150 MB instead of ~1 GB.
- **`USER appuser`:** if someone breaks in, they don't get root.
- **`--host 0.0.0.0` inside the container:** a container has its own `127.0.0.1`. The `docker run -p` flag decides what's actually exposed.

> **⚠ Problem we hit: `No matching distribution found for flit_core<4,>=3.11`**
> The `python:3.10-slim` image has an old pip (23.x) that mishandles PyTorch's package index (`jinja2` vs `Jinja2`), discards the ready-made package, and tries to build from source. Fix: `RUN pip install --upgrade pip` first (step 0 above). Lesson: **the image isn't your machine**, so every tool, even pip, must be set up inside it.

> **⚠ Problem we hit: container crashed with `FileNotFoundError: /app/shell-buddy-merged/model.safetensors`**
> The file existed, but with permissions `-rw-------` owned by root, and the server ran as `appuser`. The library reported "not found" instead of "permission denied". Fix: create the user **before** copying and use `COPY --chown=appuser:appuser` (steps 3–4 above). Don't `chmod` in a later layer: that stores a second 1 GB copy.
> **Debugging tip:** look inside the image yourself:
> ```bash
> docker run --rm --entrypoint ls shell-buddy:v2 -la /app/shell-buddy-merged
> ```

### Step 8.5: Build

```bash
cd ~/projects/model_build
docker build -t shell-buddy:v2 .
docker images shell-buddy        # ~3.39 GB on disk
```
`-t name:tag` names the image; the tag matches the model version. `transferring context: ~1 GB` is the model. If it's several GB, `.dockerignore` isn't working.

### Step 8.6: Run

Stop any uvicorn from Lesson 7 first (both use port 8765).
```bash
docker run -d --name shell-buddy -p 127.0.0.1:8765:8765 shell-buddy:v2
docker logs -f shell-buddy      # wait for "Application startup complete", then Ctrl+C (container keeps running)
docker ps                       # STATUS shows "(healthy)" after ~1 minute
```

| Flag | Meaning |
|---|---|
| `-d` | Run in the background |
| `--name shell-buddy` | A name to refer to it |
| `-p 127.0.0.1:8765:8765` | Laptop's 127.0.0.1:8765 → container's 8765. **Only your laptop** can reach it. |

If a container with that name already exists (e.g. after a crash): `docker rm shell-buddy` first.

### Step 8.7: Test and manage

```bash
curl -s http://127.0.0.1:8765/health
howto "find all png files in this folder"     # works unchanged: same API, different setup behind it
```

```bash
docker stop shell-buddy          # stop (frees ~3 GB RAM)
docker start shell-buddy         # start again (~10 s to load)
docker rm -f shell-buddy         # delete the container (image stays)
docker rmi shell-buddy:v2        # delete the image (frees ~3.4 GB)
```

### Is it "deployed" now?

Deployment has levels:
```
Level 0  Packaged      The image exists                          ✅ docker build
Level 1  Local         Running as a service on your laptop       ✅ here after Lesson 8
Level 2  Network       Other machines on your Wi-Fi/LAN can use it
Level 3  Internet      Anyone, anywhere (cloud server + HTTPS + login)   ✅ after AWS Lesson E
```

**Running it on another machine:** `docker save shell-buddy:v2 | gzip > shell-buddy-v2.tar.gz` → copy → `docker load < shell-buddy-v2.tar.gz`, or push to a **private** registry. (Image is x86_64; ARM Macs need a separate build.)

**Letting others use your server:** never expose it (`-p 8765:8765`, `0.0.0.0`) without an **API key, HTTPS and a rate limit**. On an office network, ask IT first. AWS Lesson E does this properly.

---

## Lesson 9: Quantization with Ollama

### What and why

The model is ~500 million numbers stored precisely. **Quantization rounds them** (like 3.14 instead of 3.14159265): smaller, faster, slightly less accurate.

| | Before | After (8-bit) |
|---|---|---|
| File size | ~1 GB | **531 MB** |
| RAM while running | ~3 GB | **616 MB** |
| Time per request | ~1 s via server | **~1.1 s, on the laptop GPU** |
| Accuracy | — | Lost ~1 example in 100 |

**Two new tools:**
- **GGUF**: a model **file format** (like saving Word as PDF: same content, different program opens it).
- **Ollama**: a **program that runs models** from GGUF files, fast on normal hardware. It replaces PyTorch + FastAPI and has its own API on **port 11434**.

### Step 9.1: Install Ollama (read the script first)

```bash
cd ~
curl -fsSL https://ollama.com/install.sh -o ollama-install.sh
less ollama-install.sh        # read it; q to quit
sh ollama-install.sh
ollama --version
systemctl status ollama --no-pager    # "active (running)"
```
Ollama now runs in the background on `127.0.0.1:11434` (this computer only).

### Step 9.2: Get the converter (llama.cpp)

```bash
cd ~/projects
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd ~/projects/model_build
source .mb/bin/activate
pip install ~/projects/llama.cpp/gguf-py sentencepiece
```
`--depth 1` skips the full history. **Don't** install llama.cpp's full requirements file: it pins its own torch/transformers versions and could break your setup.

### Step 9.3: Convert and quantize to 8-bit in one step

```bash
python ~/projects/llama.cpp/convert_hf_to_gguf.py shell-buddy-merged \
  --outfile shell-buddy-q8_0.gguf --outtype q8_0
ls -lh shell-buddy-q8_0.gguf      # ~531 MB
```
`q8_0` = 8 bits per number (half of 16). For a small 0.5B model, 8-bit is safer than 4-bit, which loses more accuracy on small models.

> **⚠ Problem we hit (twice, my wrong instructions):**
> 1. `ollama create ... --quantize q4_K_M` on a GGUF file → `create-time quantization is only supported for safetensors imports; quantize GGUF models before importing`.
> 2. Pointing it at the safetensors folder instead → `unsupported --quantize "q4_K_M": supported types are int4, int8, nvfp4, mxfp4, mxfp8` (those are Apple/MLX formats).
>
> Ollama 0.40's `--quantize` doesn't do llama.cpp-style formats. **Quantize with llama.cpp first, then import the finished file without `--quantize`** (Steps 9.3–9.5). Sources: [Ollama docs: Importing a model](https://docs.ollama.com/import), [jacar.es guide](https://jacar.es/en/import-hugging-face-model-into-ollama/).

### Step 9.4: The Modelfile

A **Modelfile** tells Ollama which file to use and how to talk to it. Create `Modelfile` (no extension):

```text
FROM ./shell-buddy-q8_0.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are Shell Buddy. Convert the user's request into a single Linux bash command. Reply with only the command, no explanation and no markdown."""

PARAMETER temperature 0
PARAMETER num_predict 80
PARAMETER stop "<|im_end|>"
```

| Line | Tells Ollama | Why it matters |
|---|---|---|
| `FROM` | Which model file | Your GGUF |
| `TEMPLATE` | The chat format | Must match training (`<\|im_start\|>` …). `{{ .System }}` and `{{ .Prompt }}` are blanks Ollama fills in. |
| `SYSTEM` | The system prompt | Must be **exactly** the training text |
| `PARAMETER` | Settings | `temperature 0` = always the most likely answer; `stop` = where an answer ends |

### Step 9.5: Import and try it

```bash
cd ~/projects/model_build
ollama create shell-buddy -f Modelfile
ollama list                          # shell-buddy:latest  531 MB
rm -f shell-buddy-f16.gguf           # if you made a 16-bit file earlier, it's no longer needed

ollama run shell-buddy "find all files bigger than 1GB in my home folder"
time ollama run shell-buddy "count lines in all python files in this directory"
ollama ps
```
`ollama run "..."` asks once and exits (without quotes it opens a chat; `/bye` to leave). It only **prints** commands, never runs them.

**What `ollama ps` showed:** `616 MB, 100% GPU`. That's the **MX230** we gave up on in Lesson 1: too old for PyTorch training, but llama.cpp still supports it, and a 616 MB model fits in 2 GB. **A smaller model fits on smaller hardware.** `UNTIL 4 minutes from now`: Ollama unloads a model 5 minutes after its last use.

### Step 9.6: Did rounding make it worse?

Create `eval_ollama.py` (same scoring as `evaluate.py`, but asks Ollama over HTTP):

```python
import json
import re
import subprocess
import sys
import urllib.request
from collections import Counter

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MODEL = sys.argv[2] if len(sys.argv) > 2 else "shell-buddy"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
SYSTEM_PROMPT = (
    "You are Shell Buddy. Convert the user's request into a single Linux bash command. "
    "Reply with only the command, no explanation and no markdown."
)


def ask_ollama(request):
    body = json.dumps({
        "model": MODEL,
        "system": SYSTEM_PROMPT,     # same prompt as training, so any model gets the same instructions
        "prompt": request,
        "stream": False,             # wait for the full answer instead of word by word
        "options": {"temperature": 0, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["response"].strip()


# Same scoring rules as evaluate.py
def extract_command(text):
    match = re.search(r"```(?:bash|sh)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1)
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def clean_format(text):
    return "```" not in text and "\n" not in text


def normalize(cmd):
    return " ".join(cmd.replace('"', "'").split())


def valid_syntax(cmd):
    return subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, timeout=5).returncode == 0


with open("data/test.jsonl") as f:
    rows = [json.loads(line) for line in f][:N]

scores = Counter()
for i, row in enumerate(rows, start=1):
    request = row["messages"][1]["content"]
    expected = row["messages"][2]["content"]
    out = ask_ollama(request)
    cmd = extract_command(out)
    scores["clean_format"] += clean_format(out)
    scores["valid_syntax"] += bool(cmd) and valid_syntax(cmd)
    scores["same_program"] += bool(cmd) and cmd.split()[0] == expected.split()[0]
    scores["exact_match"] += normalize(cmd) == normalize(expected)
    print(f"Scored {i}/{len(rows)}", end="\r", flush=True)

print(f"\n\nModel: {MODEL}")
for metric in ("clean_format", "valid_syntax", "same_program", "exact_match"):
    print(f"{metric:<15}{scores[metric] / len(rows):>6.0%}")
```

```bash
python eval_ollama.py                    # ~2 minutes
python eval_ollama.py 100 qwen2.5:7b     # optional: big generalist, ~10–20 min on CPU
```

| | v2 16-bit | **Ollama 8-bit** | qwen2.5:7b (not trained) |
|---|---|---|---|
| clean_format | 100% | 100% | 99% |
| valid_syntax | 100% | 99% | 100% |
| same_program | 92% | **91%** | 63% |
| exact_match | 19% | **18%** | 9% |
| Size | ~1 GB | **531 MB** | 4.7 GB |

**Two lessons:** 8-bit rounding cost ~1 example in 100. And **a small model fine-tuned for one task beats a 14× bigger general model at that task**, while being cheaper and faster. That's the main reason companies fine-tune. (Caveat: the score rewards this dataset's style; the 7B model is probably better at everyday commands.)

### Step 9.7: Test the safety warnings automatically

> **⚠ Problem we hit: no warning for `rmdir`**
> `find /tmp -type d -empty -exec rmdir {} \;` ran without a warning, because the pattern `\brm\b` only matches `rm` as a separate word. (Harmless: `rmdir` only removes empty folders; Linux blocked the system ones.) I hadn't tested my own pattern. Fix: `(r"\b(rm|rmdir|unlink|shred)\b|-delete\b", "deletes files")`, and a test file so it can't silently break again.

Create `test_howto.py`:
```python
from howto import danger_warnings

# command → should it show a warning?
cases = {
    "find /tmp -type d -empty -exec rmdir {} \\;": True,
    "rm -rf build": True,
    "find . -name '*.log' -delete": True,
    "echo '* * * * * date' | crontab": True,
    "sudo apt update": True,
    "ls -la": False,
    "find . -name '*.png'": False,
    "df -h": False,
}

for cmd, should_warn in cases.items():
    warned = bool(danger_warnings(cmd))
    print("OK  " if warned == should_warn else "FAIL", cmd)
```
```bash
cd ~/projects/model_build
.mb/bin/python test_howto.py      # every line must say OK
```
Importing `howto` is safe because `main()` only runs from the command line.

### Step 9.8: Switch `howto` to Ollama

Only `suggest_via_api()` changes; guardrails stay the same.

1. Replace the `API_URL` line with:
   ```python
   API_URL = os.environ.get("SHELL_BUDDY_URL", "http://127.0.0.1:11434")   # Ollama's port
   OLLAMA_MODEL = os.environ.get("SHELL_BUDDY_MODEL", "shell-buddy")
   ```
2. Replace `suggest_via_api` with:
   ```python
   def suggest_via_api(request):
       """Ask Ollama for a command. Returns None if Ollama isn't reachable."""
       body = json.dumps({
           "model": OLLAMA_MODEL,
           "system": SYSTEM_PROMPT,
           "prompt": request,
           "stream": False,
           "options": {"temperature": 0, "num_predict": 80},
       }).encode()
       req = urllib.request.Request(f"{API_URL}/api/generate", data=body, headers={"Content-Type": "application/json"})
       try:
           with urllib.request.urlopen(req, timeout=60) as resp:
               text = json.loads(resp.read())["response"].strip()
       except (urllib.error.URLError, TimeoutError):
           return None
       return text.splitlines()[0].strip() if text else ""
   ```
3. In `main()`, change the fallback message to `"(Ollama not reachable, loading the model locally...)"`.
4. Make sure no old `SHELL_BUDDY_URL` overrides the new default:
   ```bash
   echo $SHELL_BUDDY_URL               # should print an empty line
   grep -n SHELL_BUDDY ~/.bashrc       # no output = not set
   unset SHELL_BUDDY_URL               # clears it from the current terminal if needed
   ```
5. Stop Docker and test:
   ```bash
   docker stop shell-buddy
   time howto "find all png files in this folder"     # user time ~0.2 s
   ollama ps
   ```

Ollama's API is general-purpose (any program, any prompt, no length limit). That's fine while it only listens on 127.0.0.1. To share it, put a gateway in front of it (AWS Lesson E).

**Sharing the model by file:** someone can run their own copy with just `shell-buddy-q8_0.gguf`, `Modelfile` and `howto.py`: install Ollama, `ollama create shell-buddy -f Modelfile`, set up the alias. No Python libraries, PyTorch or Docker needed.

---

# Part 3: Save it

## GitHub lesson: upload your code

### Why

- **Backup:** if your laptop breaks, the code is safe.
- **History:** every change is saved; you can go back.
- **Easy download anywhere** (`git clone`).

GitHub **stores** code; it can't run an always-on server like Ollama (Pages = static websites, Actions = short jobs). **GitHub holds the code, AWS runs it.**

**Git vs. GitHub:** Git is a program on your laptop that saves **snapshots** (commits). GitHub is a website that stores a copy. **Push** = upload your snapshots.

### What to upload

**Upload the code that produces things, not the things it produces.**

| Upload ✅ | Don't upload ❌ | Why not |
|---|---|---|
| `explore_data.py`, `clean_data.py`, `inspect_data.py`, `run_base.py` | `.mb/` (1.4 GB) | Anyone can recreate it with `pip install` |
| `evaluate.py`, `eval_ollama.py`, `merge_model.py` | `data/` | Dataset license unclear; `clean_data.py` recreates it |
| `howto.py`, `server.py`, `gateway.py`, `test_howto.py` | `shell-buddy-merged/`, `*.gguf` | GitHub rejects files over 100 MB |
| `Dockerfile`, `.dockerignore`, `requirements-server.txt`, `Modelfile` | `shell-buddy-lora*/` | Recreated by training |
| `shell_buddy_training.ipynb`, `README.md` | Passwords, API keys | **Never** put secrets in code |

### Step G1: `.gitignore`

```text
# Python environment and cache
.mb/
__pycache__/

# Data (license unclear; recreate with clean_data.py)
data/

# Models (too big for GitHub)
shell-buddy-merged/
shell-buddy-lora/
shell-buddy-lora-v2/
*.gguf
*.safetensors
*.zip

# Local tool settings
.claude/
```

Before uploading the notebook, open it and check it contains no passwords or tokens.

### Step G2: Tell Git who you are

```bash
git config --global user.name
git config --global user.email
# if empty:
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

### Step G3: SSH key (your login for GitHub)

GitHub doesn't accept your password from the terminal. You use an **SSH key pair**:
- **Private key** (`id_ed25519`): your house key. **Never share it.**
- **Public key** (`id_ed25519.pub`): the lock. You give it to GitHub.

```bash
ls ~/.ssh/id_ed25519.pub                 # already have one?
ssh-keygen -t ed25519 -C "github"        # if not; Enter for the default location
cat ~/.ssh/id_ed25519.pub                # copy this whole line
```
GitHub → profile picture → **Settings → SSH and GPG keys → New SSH key** → paste → **Add SSH key**.
```bash
ssh -T git@github.com                    # "Hi <your-username>! You've successfully authenticated"
```

### Step G4: Create the repository

GitHub → **+ → New repository** → name `shell_buddy` → **Private** → don't add README/.gitignore/license → **Create repository**.

### Step G5: Upload

```bash
cd ~/projects/model_build
git init -b main
git add .
git status
```
**Read `git status` before continuing.** Only small files should be listed. If you see `.mb/`, `data/`, `.gguf` or `shell-buddy-merged`, stop.

```bash
git commit -m "Shell Buddy: fine-tuned English-to-bash model"
git remote add origin git@github.com:<your-username>/shell_buddy.git
git push -u origin main
```

| Command | Meaning |
|---|---|
| `git init -b main` | Start tracking this folder |
| `git add .` | Choose everything (except `.gitignore` entries) for the next snapshot |
| `git commit -m "..."` | Take the snapshot |
| `git remote add origin ...` | Where the GitHub copy lives |
| `git push -u origin main` | Upload. `-u` remembers the destination, so next time just `git push`. |

> **⚠ Problem we hit: `Permission to <personal>/shell_buddy.git denied to <work-account>` (403)**
> Two GitHub accounts on one laptop. The remote used `https://`, so Git used the **saved work-account login**. Fix: switch to SSH, so the **key** decides the account:
> ```bash
> git remote set-url origin git@github.com:<personal-username>/shell_buddy.git
> ssh -T git@github.com      # which account does your key belong to?
> ```
> If the key belongs to the work account, make a second key for the personal one (a key can only belong to one account):
> ```bash
> ssh-keygen -t ed25519 -C "personal-github" -f ~/.ssh/id_ed25519_personal
> cat ~/.ssh/id_ed25519_personal.pub      # add to the PERSONAL account on GitHub
> nano ~/.ssh/config
> ```
> ```text
> Host github-personal
>     HostName github.com
>     User git
>     IdentityFile ~/.ssh/id_ed25519_personal
>     IdentitiesOnly yes
> ```
> ```bash
> git remote set-url origin git@github-personal:<personal-username>/shell_buddy.git
> ssh -T git@github-personal          # Hi <personal-username>!
> git push -u origin main
> ```
> To use your personal email for **this repo only**: `git config user.email "personal@example.com"` then `git commit --amend --reset-author --no-edit` (safe only before pushing).

### Step G6: README

`README.md` is the front page of your repository. Create it:

````markdown
# Shell Buddy

A small AI model that turns plain English into Linux commands.

```
$ howto "find all files bigger than 1GB in my home folder"

  find ~ -size +1G

Run it? [y/N]
```

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
````

The everyday loop from now on: **edit → check → commit → push**.
```bash
git status
git add README.md
git commit -m "Add README"
git push
```

---

# Part 4: Put it in the cloud

**AWS (Amazon Web Services)** rents computers in Amazon's data centers by the hour. The rented computer is an **EC2 instance**: a second laptop that lives in Amazon's building, is always on, and has a public internet address.

```
Lesson A   Account + safety (surprise bills, hackers)
Lesson B   Rent a Linux computer and connect from your terminal
Lesson C   Install Ollama on it and copy your model there
Lesson D   Use it from your laptop through a private tunnel (only you)
Lesson E   Let others use it safely (API key + HTTPS)
Lesson F   Switch off / clean up so you stop paying
```

**Deploy first, improve later:** the cloud setup doesn't depend on the model version. Updating to v3 later = copy a new GGUF and run `ollama create` again (~5 minutes).

**Anything on the internet gets attacked.** Bots scan every public address within minutes, so we keep every door closed unless it's needed and protected.

---

## AWS Lesson A: Account and safety

### Why

Two protections **before renting anything**: an email alert on spending, and a second login step (MFA). Nothing here costs money.

**Never share** your AWS password, MFA codes, card details, or any "access key" with anyone, including in chats. Only log in on `aws.amazon.com`.

### Step A1: Sign up

1. **[aws.amazon.com](https://aws.amazon.com) → Create an AWS account**
2. Personal email and an account name; enter the email verification code
3. Contact info: **Personal**; phone verification
4. If asked to choose **Free plan vs Paid plan, choose Free**: you use free credits and can't be charged beyond them.

> **Note: two kinds of AWS account.** Our account was made with AWS's **new sign-up flow** (Builder ID login, "projects", Free plan). In this type:
> - Your **Builder ID login with MFA is your everyday login**. You **don't** need to create an IAM admin user. (That advice is for the classic "root user" account type; I gave those steps first by mistake.)
> - The **region is chosen for you**.
> - Ignore "AWS Global View" / "Local Zones" and **don't "upgrade"** (that switches to the advanced type with more setup).
>
> Source: [Compare sign-up options (AWS docs)](https://docs.aws.amazon.com/accounts/latest/reference/sign-up-for-aws.html)

### Step A2: Turn on MFA

**MFA** = password **plus** a 6-digit code from your phone. A stolen password alone isn't enough.

1. Install **Google Authenticator** or **Microsoft Authenticator** on your phone
2. AWS → your name (top right) → **Security credentials** → **Assign MFA device**
3. Name it (`my-phone`), choose **Authenticator app**, scan the QR code
4. Enter **two codes in a row** → **Add MFA**

### Step A3: Budget alert

1. Search **Budgets** → **Create budget**
2. **Use a template (simplified) → Monthly cost budget**
3. Amount: we used **$0.01**, so you get an email the moment anything costs a cent
4. Your email → **Create budget**

An alert **doesn't stop** anything; it's an early warning.

### Step A4: Find your region

Open **EC2** (search bar). The region is shown **top right**. On Billing/Budgets/IAM pages it says "Global", because those apply to the whole account.

Ours: **Asia Pacific (Sydney)**, `ap-southeast-2`. From India it adds ~0.15 s per request, which you won't notice.

**If you ever can't find your server, check the region first.** Resources only show in the region they were created in.

---

## AWS Lesson B: Rent a cloud computer

### Why

Shell Buddy needs a computer that's always on. We'll rent one and connect with **SSH**, the same secure tool Git uses.

### Step B1: Make a key just for this server

SSH logs in with a key pair. The **public** half goes on the server; the **private** half never leaves your laptop.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/aws_shellbuddy -C "aws-shellbuddy"
```
Then: EC2 → **Network & Security → Key Pairs → Actions → Import key pair** → name `shell-buddy-key` → **Browse** → `~/.ssh/aws_shellbuddy.pub` (Ctrl+H shows hidden folders) → **Import**.

**Only the `.pub` file goes to AWS.**

### Step B2: Launch the instance

**EC2 Dashboard → Launch instance:**

| Field | Value | What it means |
|---|---|---|
| Name | `shell-buddy` | A label |
| OS image (AMI) | **Ubuntu Server 24.04 LTS**, 64-bit (x86) | The operating system it starts with |
| Instance type | **c7i-flex.large** (2 CPUs, 4 GB RAM), must say **"Free tier eligible"** | The size of the computer |
| Key pair | `shell-buddy-key` | The lock matching your key |
| Network settings | **Allow SSH traffic from → My IP**. HTTP/HTTPS **unticked**. | The **security group** (the server's firewall) |
| Storage | **20 GiB** gp3 | The disk (Ollama + model ≈ 5 GB) |

→ **Launch instance**. Cost ~$0.10/hour from free credits.

**Why "My IP" and not "Anywhere"?** Bots knock on port 22 all day. Your key makes it hard to get in; "My IP" means their knocks never reach the server at all. A good lock **and** a hidden door.

**Why no HTTP/HTTPS yet?** Nothing is running there yet, and Ollama has **no password**: if exposed, anyone could use your server with your credits. We open HTTPS only in Lesson E, after adding an API key and encryption.

### Step B3: Connect

1. **Instances** → wait for **Running** and **2/2 checks passed**
2. Copy the **Public IPv4 address**
3. On your laptop:
   ```bash
   IP=PASTE_IP_HERE
   ssh -i ~/.ssh/aws_shellbuddy ubuntu@$IP
   ```
   First time: "Are you sure you want to continue connecting?" → `yes` (saves the server's fingerprint so SSH can warn you if a different machine pretends to be it).
4. Your prompt becomes `ubuntu@ip-...`. Check the machine:
   ```bash
   uname -a; nproc; free -h; df -h /
   ```
   Ours: 2 CPUs, 3.7 GB RAM, 17 GB free.
5. `exit` returns to your laptop.

**Two habits:**
- If SSH stops connecting one day, your home IP probably changed: update the SSH rule in the security group to "My IP" again.
- When done: **Instance state → Stop**. **Never Terminate** (deletes the server). Without an Elastic IP (Lesson E), a stopped server gets a **new IP** when started again.

---

## AWS Lesson C: Put Shell Buddy on the server

### Why

Repeat what you did on your laptop, on the server: update → install Ollama → copy the model → `ollama create`.

You use **two terminals**: **Terminal S** (logged in with `ssh`, prompt `ubuntu@ip-...`) and **Terminal L** (your laptop).

### Step C1: Update the server (Terminal S)

```bash
sudo apt update && sudo apt upgrade -y
```
`apt` is Ubuntu's command-line app store. If a purple screen asks which services to restart, press **Enter**. If it says **"System restart required"**: `sudo reboot`, wait a minute, reconnect.

### Step C2: Install Ollama (Terminal S)

```bash
curl -fsSL https://ollama.com/install.sh -o install.sh
less install.sh        # read it; q to quit
sh install.sh
ollama --version
```
"No NVIDIA/AMD GPU detected… CPU-only mode" is expected. Ollama runs as a background **service** on `127.0.0.1:11434`, reachable only from inside the server.

### Step C3: Copy the model (Terminal L)

`scp` = copy over SSH (same key, same firewall door):
```bash
IP=PASTE_IP_HERE
cd ~/projects/model_build
scp -i ~/.ssh/aws_shellbuddy shell-buddy-q8_0.gguf Modelfile ubuntu@$IP:~/
```
531 MB may take a few minutes.

### Step C4: Create and test (Terminal S)

```bash
ls -lh ~
ollama create shell-buddy -f Modelfile
ollama list
time ollama run shell-buddy "show disk usage in human readable format"     # df -h, ~1.7 s
ollama ps
```

(Ours showed `29%/71% CPU/GPU` on a server with no graphics card. Probably Ollama counting a software graphics driver as a GPU; harmless. To check: `journalctl -u ollama --no-pager | grep -i -E "vulkan|gpu|compute" | tail -8`.)

---

## AWS Lesson D: Use it privately from your laptop

### Why `howto.py` stays on your laptop

Shell Buddy has two parts:

| Part | Job | Where |
|---|---|---|
| The model (the brain) | Turns English into a command; big and heavy | **Cloud** |
| `howto.py` (the hands) | Asks the brain, shows the command, checks danger, asks yes/no, **runs it** | **Your laptop** |

`howto` must run commands **on the computer you're using**: "find png files in this folder" means **your** files. Same as the ChatGPT app: a small app on your phone, a big model in a data center. Bonus: one cloud brain can serve many laptops, and a model update happens once on the server.

### Step D1: Open an SSH tunnel (Terminal L)

An **SSH tunnel** is a private pipe inside your SSH connection. Ollama stays hidden from the internet.
```
howto → 127.0.0.1:11435 (laptop) ══SSH tunnel══► 127.0.0.1:11434 (server's Ollama)
```
We use **11435** on the laptop because your local Ollama already uses 11434.
```bash
ssh -i ~/.ssh/aws_shellbuddy -N -L 11435:127.0.0.1:11434 ubuntu@$IP
```
- `-L 11435:127.0.0.1:11434` = send laptop port 11435 to server port 11434
- `-N` = no shell, just keep the tunnel open

It **looks stuck** with no output. That's correct. Leave it open.

### Step D2: Test the tunnel (new laptop terminal)

```bash
curl -s http://127.0.0.1:11435/api/tags      # lists shell-buddy:latest, answered from Sydney
```

### Step D3: Point `howto` at the cloud

```bash
SHELL_BUDDY_URL=http://127.0.0.1:11435 $HOME/projects/model_build/.mb/bin/python $HOME/projects/model_build/howto.py "find all png files in this folder"
```
The cloud model only **suggests**. If you say yes, the command runs **on your laptop**.

> **⚠ Problem we hit: a typo made it silently use the laptop**
> `SHELLSHELL_BUDDY_URL=...` isn't the variable `howto.py` reads, so it fell back to its default (`127.0.0.1:11434`, the laptop's own Ollama). The answer looked right, but came from the wrong machine.

### Step D4: Prove it used the cloud

A test that can only give one answer:
1. **Server:** `ollama stop shell-buddy` then `ollama ps` → empty
2. **Laptop:** run the D3 command (answer n)
3. **Server:** `ollama ps` → model loaded again (only your tunnel can reach it, so your request loaded it). Log check: `sudo journalctl -u ollama --since "15 minutes ago" --no-pager | grep "api/generate"`
4. **Laptop:** `ollama ps` → empty (your local Ollama wasn't used)

### Step D5: A nickname for the server

The **SSH config file** stores connection details under a nickname:
```bash
nano ~/.ssh/config
```
Add at the bottom (the server's **public** IP, not your laptop's and not the private `172.31...` one):
```text
Host shellbuddy
    HostName PASTE_SERVER_PUBLIC_IP
    User ubuntu
    IdentityFile ~/.ssh/aws_shellbuddy
    IdentitiesOnly yes
```
Now:
```bash
ssh shellbuddy                                  # log in
ssh -N -L 11435:127.0.0.1:11434 shellbuddy      # open the tunnel
```
After every server restart, update `HostName` (until you add an Elastic IP in Lesson E).

### Step D6: A `howto-cloud` command (tunnel version)

Add to `~/.bashrc` below the `howto` alias, then `source ~/.bashrc`:
```bash
alias howto-cloud='SHELL_BUDDY_URL=http://127.0.0.1:11435 $HOME/projects/model_build/.mb/bin/python $HOME/projects/model_build/howto.py'
```
`howto` = laptop Ollama; `howto-cloud` = cloud (tunnel must be open). In Lesson E this alias is replaced by the public HTTPS version.

---

## AWS Lesson E: Let other people use it safely

### The idea

To let others in without giving them your SSH key, we add a **front desk** with three layers:

```
Friend's laptop ──HTTPS──► Caddy ──► gateway.py ──► Ollama (shell-buddy)
                         (port 443)  (checks key)   (stays hidden)
```

| Layer | Job | Analogy |
|---|---|---|
| **Caddy** | Encrypts traffic (HTTPS), so nobody can read requests or steal keys on the way | Armored delivery truck |
| **gateway.py** | Checks the **API key**, limits each key to 20 requests/minute, allows only "suggest a command" | Receptionist checking passes |
| **Ollama** | Runs the model; unreachable from outside | Expert in the back room |

Why not open Ollama directly? It has **no password** and lets anyone download or delete models and use your server. The gateway allows **one action only**.

### Step E0: A permanent address (Elastic IP)

The public IP changes on every restart. An **Elastic IP** is a fixed address AWS reserves for you.
1. EC2 → **Network & Security → Elastic IPs → Allocate Elastic IP address → Allocate**
2. Select it → **Actions → Associate Elastic IP address** → instance `shell-buddy` → **Associate**
3. Put it in `HostName` in `~/.ssh/config`. **It never changes again.**

Cost ~$0.005/hour from credits. **Release** it if you stop using Shell Buddy for good (Lesson F).

### Step E1: The gateway (write it on your laptop)

Create `gateway.py` in `~/projects/model_build`:

```python
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
```

- **Keys are not in the code.** They come from a protected file on the server, so the code is safe on GitHub.
- **No keys set → every request refused.** If misconfigured, it fails **closed** (locked), not open.
- Danger warnings and the yes/no prompt stay in `howto` on the user's laptop, where the command runs.

Copy it to the server:
```bash
ssh shellbuddy "mkdir -p ~/gateway"
scp gateway.py shellbuddy:~/gateway/
```

### Step E2: Install what it needs (Terminal S, `ssh shellbuddy`)

```bash
sudo apt install -y python3-venv
cd ~/gateway
python3 -m venv venv
venv/bin/pip install fastapi==0.162.0 uvicorn==0.54.0
```

### Step E3: Create an API key (Terminal S)

A long random string works as a password for programs. It's stored in a file only root can read:
```bash
KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "SHELL_BUDDY_KEYS=$KEY" | sudo tee /etc/shell-buddy.env > /dev/null
sudo chmod 600 /etc/shell-buddy.env
echo $KEY
```
Save the printed key in a password manager. **Don't paste it into chats.** If you lose it: `sudo cat /etc/shell-buddy.env` (the part after `=`).

### Step E4: Run the gateway as a service (Terminal S)

A **systemd service** keeps a program running in the background, restarts it if it crashes, and starts it on boot (Ollama runs this way too).
```bash
sudo nano /etc/systemd/system/shell-buddy.service
```
```ini
[Unit]
Description=Shell Buddy API gateway
After=network.target ollama.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/gateway
EnvironmentFile=/etc/shell-buddy.env
ExecStart=/home/ubuntu/gateway/venv/bin/uvicorn gateway:app --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
Save (**Ctrl+O, Enter, Ctrl+X**), then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now shell-buddy
systemctl status shell-buddy --no-pager
```

**What each line means:**

| Line | Meaning |
|---|---|
| `Description` | Human-readable name in status output |
| `After=network.target ollama.service` | Start order: after the network and Ollama |
| `User=ubuntu` | Run as the normal user, not root (less power = less damage if hacked) |
| `WorkingDirectory` | Start in this folder, so `gateway:app` finds `gateway.py` |
| `EnvironmentFile` | systemd (running as root) reads the locked key file and passes `SHELL_BUDDY_KEYS` to the program. The program gets the key; the file stays locked. |
| `ExecStart` | The exact command, with full paths (systemd doesn't know your PATH or venv). `--host 127.0.0.1` keeps the gateway **private**: only Caddy talks to it. |
| `Restart=on-failure` | Restart automatically if it crashes |
| `WantedBy=multi-user.target` | When enabled, start during normal boot |

| Command | What it does |
|---|---|
| `sudo systemctl daemon-reload` | Make systemd **re-read** service files. Run after every `.service` edit. |
| `sudo systemctl enable --now shell-buddy` | **`enable`** = start on every boot; **`--now`** = also start right now |
| `systemctl status shell-buddy --no-pager` | Health report: running?, since when, memory, last log lines |
| `sudo systemctl restart shell-buddy` | **After changing `gateway.py` or the key file** (they're only read at start) |
| `sudo systemctl stop / disable shell-buddy` | Stop now / don't start on boot |
| `journalctl -u shell-buddy -f` | Watch its log live (Ctrl+C to quit) |

### Step E5: Test privately on the server (Terminal S, same terminal where `$KEY` is set)

```bash
curl -s http://127.0.0.1:8000/health; echo
curl -s -X POST http://127.0.0.1:8000/suggest -H "Content-Type: application/json" -d '{"request":"show free memory"}'; echo
curl -s -X POST http://127.0.0.1:8000/suggest -H "Content-Type: application/json" -H "X-API-Key: wrongkey" -d '{"request":"show free memory"}'; echo
curl -s -X POST http://127.0.0.1:8000/suggest -H "Content-Type: application/json" -H "X-API-Key: $KEY" -d '{"request":"show free memory"}'; echo
```
Expected: `{"status":"ok"}`, `invalid API key`, `invalid API key`, `{"command":"free -m"}`.

### Step E6: Open the firewall for HTTPS

**HTTPS** encrypts everything between a user and your server. It needs a **certificate** (a digital ID card) from **Let's Encrypt** (free), which **Caddy** fetches and renews automatically. A certificate is issued for a **name**, not an IP, so we use **sslip.io**: `<ELASTIC_IP>.sslip.io` automatically points to that IP (dots or dashes both work; ours is `16.178.171.116.sslip.io`).

EC2 → Instances → `shell-buddy` → **Security** tab → security group link → **Edit inbound rules → Add rule** twice:
- **HTTPS** (443), Source **Anywhere-IPv4**
- **HTTP** (80), Source **Anywhere-IPv4** (for Let's Encrypt's check and the http→https redirect)

**Keep SSH as "My IP".** Ports 8000 and 11434 stay **closed**.

### Step E7: Install Caddy (Terminal S)

```bash
sudo apt install -y caddy
```

### Step E8: Configure Caddy (Terminal S)

A **reverse proxy** receives traffic from outside and forwards it to a program inside.
```bash
sudo nano /etc/caddy/Caddyfile
```
Delete everything (hold **Ctrl+K**), paste, and replace the name with yours:
```text
<ELASTIC_IP>.sslip.io {
    request_body {
        max_size 10KB
    }
    reverse_proxy 127.0.0.1:8000
}
```
- `<name> { ... }`: get a certificate for this name and serve HTTPS
- `max_size 10KB`: reject huge uploads (a valid request is under 1 KB)
- `reverse_proxy 127.0.0.1:8000`: pass requests to the gateway

```bash
sudo systemctl reload caddy
sleep 15
journalctl -u caddy --no-pager --since "2 minutes ago" | grep -i -E "certificate obtained|error" | tail -5
```
Look for **"certificate obtained successfully"**.

### Step E9: Store your key on the laptop

The **API key** from Step E3 (not your SSH key). A file only you can read keeps it out of your terminal history:
```bash
nano ~/.shell_buddy_key          # paste the key on one line, save
chmod 600 ~/.shell_buddy_key
```

| Key | Job | Who gets it |
|---|---|---|
| SSH key (`aws_shellbuddy`) | Logs in to the server as admin | **Only you, ever** |
| API key | Lets someone ask Shell Buddy for a command, nothing else | You, and later friends get their own |

### Step E10: Test over the internet (Terminal L)

```bash
HOST=<ELASTIC_IP>.sslip.io
curl -s https://$HOST/health; echo
curl -s -X POST https://$HOST/suggest -H "Content-Type: application/json" -d '{"request":"show free memory"}'; echo
curl -s -X POST https://$HOST/suggest -H "Content-Type: application/json" -H "X-API-Key: $(cat ~/.shell_buddy_key)" -d '{"request":"show free memory"}'; echo
```
Expected: `ok`, `invalid API key`, `{"command":"free -m"}`. (`curl` checks certificates by default, so getting `ok` proves HTTPS works.)

**Security check: hidden doors must stay hidden.**
```bash
curl -s -m 5 http://<ELASTIC_IP>:11434; echo " ollama exit=$?"
curl -s -m 5 http://<ELASTIC_IP>:8000;  echo " gateway exit=$?"
```
Both must **time out** (`exit=28`).

### Step E11: Teach `howto` to use the public server

`howto` now calls the public HTTPS address with your API key. This path needs **no PyTorch or model** on the user's laptop, just plain Python.

**Final `howto.py`** (complete file):

```python
"""howto: turn an English request into a bash command using the fine-tuned Shell Buddy model."""
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Path relative to THIS file, so howto works from any folder
MODEL_DIR = Path(__file__).resolve().parent / "shell-buddy-merged"
GATEWAY_URL = os.environ.get("SHELL_BUDDY_GATEWAY")     # public server, e.g. https://x.sslip.io
KEY_FILE = Path.home() / ".shell_buddy_key"            # API key for the public server

API_URL = os.environ.get("SHELL_BUDDY_URL", "http://127.0.0.1:11434")   # Ollama's port
OLLAMA_MODEL = os.environ.get("SHELL_BUDDY_MODEL", "shell-buddy")
SYSTEM_PROMPT = (
    "You are Shell Buddy. Convert the user's request into a single Linux bash command. "
    "Reply with only the command, no explanation and no markdown."
)
DANGER_PATTERNS = [
    (r"\b(rm|rmdir|unlink|shred)\b|-delete\b", "deletes files"),
    (r"\bdd\b", "writes raw data to a disk"),
    (r"\bmkfs", "formats a disk"),
    (r"\bch(mod|own)\s+-R|chmod\s+777", "changes permissions/ownership on many files"),
    (r"\|\s*crontab\b", "REPLACES your entire crontab"),
    (r"\bkill(all)?\b|\bpkill\b", "stops running processes"),
    (r"\bsudo\b", "runs as root (administrator)"),
    (r"\b(shutdown|reboot|poweroff)\b", "shuts down or restarts the machine"),
    (r"\bmv\b|sed\s+-i", "moves or modifies files"),
    (r">\s*/dev/sd", "overwrites a disk device"),
]
# These change the CURRENT shell; a child process can't do that for you
SHELL_STATE_COMMANDS = {"cd", "export", "source", ".", "alias", "unset", "pushd", "popd"}
BASH_BUILTINS = {"read", "set", "history", "ulimit", "umask", "type", "declare", "shopt", "trap", "wait"}
BOLD, YELLOW, RESET = "\033[1m", "\033[33m", "\033[0m"


def load():
    # Heavy imports live here, so the API path never pays for loading PyTorch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging

    logging.set_verbosity_error()
    logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def suggest(tokenizer, model, request):
    import torch

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text.splitlines()[0].strip() if text else ""


def suggest_via_api(request):
    """Ask Ollama for a command. Returns None if Ollama isn't reachable."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": request,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(f"{API_URL}/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = json.loads(resp.read())["response"].strip()
    except (urllib.error.URLError, TimeoutError):
        return None
    return text.splitlines()[0].strip() if text else ""


def suggest_via_gateway(request):
    """Ask the public Shell Buddy server. Exits with a clear message if it fails."""
    try:
        key = KEY_FILE.read_text().strip()
    except FileNotFoundError:
        sys.exit(f"No API key found. Save your key in {KEY_FILE}")
    body = json.dumps({"request": request}).encode()
    req = urllib.request.Request(f"{GATEWAY_URL}/suggest", data=body,
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["command"]
    except urllib.error.HTTPError as e:   # The server answered, but said no
        reasons = {401: "your API key was rejected",
                   429: "too many requests, wait a minute",
                   503: "the model on the server is unavailable"}
        sys.exit(f"Shell Buddy server refused: {reasons.get(e.code, f'error {e.code}')}")
    except (urllib.error.URLError, TimeoutError):   # Couldn't reach the server at all
        sys.exit(f"Can't reach the Shell Buddy server at {GATEWAY_URL}. Is it running?")


def danger_warnings(cmd):
    return [reason for pattern, reason in DANGER_PATTERNS if re.search(pattern, cmd)]


def first_program(cmd):
    words = cmd.split()
    if words and words[0] == "sudo" and len(words) > 1:
        words = words[1:]
    return words[0] if words else ""


def check_command(cmd):
    """Machine-specific checks. Returns (runnable, note); note explains why it isn't runnable."""
    program = first_program(cmd)
    if program in SHELL_STATE_COMMANDS:
        return False, "This changes your current shell (directory or variables). Copy it and run it yourself."
    if program not in BASH_BUILTINS and shutil.which(program) is None:
        return False, f"'{program}' isn't a program on this system, so this isn't a valid command. Not running it."
    return True, None


def main():
    if len(sys.argv) < 2:
        print('Usage: howto "describe what you want to do"')
        sys.exit(1)
    request = " ".join(sys.argv[1:])

    if GATEWAY_URL:
        cmd = suggest_via_gateway(request)
    else:
        cmd = suggest_via_api(request)
        if cmd is None:
            print("(Ollama not reachable, loading the model locally...)")
            tokenizer, model = load()
            cmd = suggest(tokenizer, model, request)

    if not cmd:
        print("Shell Buddy had no suggestion.")
        sys.exit(1)

    print(f"\n  {BOLD}{cmd}{RESET}\n")
    runnable, note = check_command(cmd)
    if not runnable:
        print(f"  {note}")
        return

    warnings = danger_warnings(cmd)
    for w in warnings:
        print(f"  {YELLOW}⚠  This command {w}{RESET}")

    try:
        if warnings:
            run = input("\nType 'yes' to run it anyway, anything else to cancel: ").strip() == "yes"
        else:
            run = input("Run it? [y/N] ").strip().lower() == "y"
    except (KeyboardInterrupt, EOFError):
        run = False
        print()

    if run:
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    else:
        print("Not run.")


if __name__ == "__main__":
    main()
```

**Three ways to get a command, in order:**
1. `SHELL_BUDDY_GATEWAY` set → public server with your API key (stops with a clear reason if it fails)
2. Otherwise → Ollama at `SHELL_BUDDY_URL` (default: laptop, 11434)
3. Ollama unreachable → load `shell-buddy-merged` locally with PyTorch

- `HTTPError` = the server **answered** with an error code; `URLError` = **couldn't connect**. `HTTPError` is a kind of `URLError`, so it must be caught **first**.

> **⚠ Problem we found: `MODEL_DIR` had gone missing** from `howto.py` (only its comment remained), so the local fallback would have crashed with `NameError`. It's restored in the file above.

### Step E12: Update the alias (Terminal L)

In `~/.bashrc`, **replace** the old `howto-cloud` line:
```bash
alias howto-cloud='SHELL_BUDDY_GATEWAY=https://<ELASTIC_IP>.sslip.io $HOME/projects/model_build/.mb/bin/python $HOME/projects/model_build/howto.py'
```
```bash
source ~/.bashrc
```
**No tunnel needed anymore.**

### Step E13: Test (Terminal L)

```bash
howto-cloud "show free memory"        # via the public server
howto "show free memory"              # via your laptop's Ollama
cd ~/projects/model_build && .mb/bin/python test_howto.py     # all 8 lines OK

# the friendly error message:
mv ~/.shell_buddy_key ~/.shell_buddy_key.bak
howto-cloud "show free memory"
mv ~/.shell_buddy_key.bak ~/.shell_buddy_key
```

### Step E14–E17: Keys for friends, and revoking them

**One key per person**, so you can cancel (**revoke**) one without affecting anyone else.

**Create (Terminal S):**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
sudo nano /etc/shell-buddy.env       # add ",newkey" at the end: SHELL_BUDDY_KEYS=yourkey,newkey  (no spaces)
sudo systemctl restart shell-buddy
```
Write down **which key belongs to whom**.

**Check it works (Terminal L):**
```bash
FRIEND=PASTE_FRIEND_KEY
curl -s -X POST https://$HOST/suggest -H "Content-Type: application/json" -H "X-API-Key: $FRIEND" -d '{"request":"show free memory"}'; echo
```

**Revoke (Terminal S):**
```bash
sudo nano /etc/shell-buddy.env       # delete ",friendkey"
sudo systemctl restart shell-buddy
```

**Confirm (Terminal L):** the same `curl` now says `invalid API key`, and `howto-cloud` still works for you.

**If a key ever leaks: revoke it and issue a new one.**

**For a real friend, send:**
1. `howto.py` (repo is private: send the file or add them as a collaborator)
2. **Their own key**, privately (direct message, not a group chat)
3. Setup lines (Linux + Python 3, nothing else):
   ```bash
   nano ~/.shell_buddy_key          # paste the key
   chmod 600 ~/.shell_buddy_key
   echo "alias howto='SHELL_BUDDY_GATEWAY=https://<ELASTIC_IP>.sslip.io python3 ~/howto.py'" >> ~/.bashrc
   source ~/.bashrc
   ```
4. A reminder: commands run **on their own laptop**, so read before typing yes. And the server only works while your instance is running.

**Save to GitHub** (no keys inside, so safe):
```bash
cd ~/projects/model_build
git status
git add gateway.py howto.py
git commit -m "Add public API gateway and gateway support in howto"
git push
```

---

## AWS Lesson F: Switching off and cleaning up

### Pause (between sessions)

EC2 → Instances → `shell-buddy` → **Instance state → Stop**. **Never Terminate** unless you mean to delete everything.

| While stopped | Cost |
|---|---|
| CPU and memory | Nothing |
| 20 GB disk (keeps Ollama, model, gateway, Caddy) | ~$2/month |
| Elastic IP (reserved, not in use) | ~$3.60/month |
| **Total** | **~$5–6/month**, from free credits |

**Coming back:** **Start instance**, wait ~1 minute. Ollama, the gateway and Caddy are all enabled systemd services, so they start by themselves, and the Elastic IP means no config changes. `howto-cloud` just works. While stopped, `howto-cloud` says *"Can't reach the Shell Buddy server"*; plain `howto` still works locally.

Check your credits sometimes: search **Credits** in the console.

### Delete (when you're done with AWS for good)

In this order:
1. **Terminate the instance** (deletes the server **and its disk**; can't be undone)
2. **Release the Elastic IP**: Elastic IPs → select → Actions → **Release**. ⚠️ The most-forgotten step: an unused Elastic IP keeps costing money.
3. Check **Volumes** and **Elastic IPs** are empty
4. (Free, but tidy) delete the security group (`launch-wizard-...`) and key pair `shell-buddy-key`
5. Check the bill a few days later

Nothing is lost: code is on GitHub, the model (`shell-buddy-q8_0.gguf`) is on your laptop.

**Tip:** save the server config files in the repo so a rebuild is copy-paste (no secrets in them):
```bash
mkdir -p ~/projects/model_build/deploy
scp shellbuddy:/etc/systemd/system/shell-buddy.service shellbuddy:/etc/caddy/Caddyfile ~/projects/model_build/deploy/
```
**Never copy `/etc/shell-buddy.env`** (that's the key file).

---

# Appendix

## Daily cheat sheet

**Start a cloud session**
```bash
# AWS console (Sydney) → EC2 → Instances → shell-buddy → Instance state → Start → wait ~1 min
howto-cloud "your request"
ssh shellbuddy                         # if you need the server's terminal
```

**End a cloud session**
```bash
# AWS console → Instance state → Stop   (NOT Terminate)
```

**Local only**
```bash
howto "your request"                   # laptop Ollama (GPU)
ollama ps                              # what's loaded
```

**Server health (Terminal S)**
```bash
systemctl status ollama shell-buddy caddy --no-pager
journalctl -u shell-buddy -f
sudo systemctl restart shell-buddy     # after editing gateway.py or the key file
```

**Update the model on the server (e.g. v3)**
```bash
scp shell-buddy-v3-q8_0.gguf Modelfile shellbuddy:~/      # Modelfile's FROM must point at the new file
ssh shellbuddy "ollama create shell-buddy:v3 -f Modelfile"
# then change MODEL = "shell-buddy:v3" in gateway.py, copy it over, restart shell-buddy
```

## Every problem we hit, in one table

| # | Where | Symptom | Cause | Fix |
|---|---|---|---|---|
| 1 | Setup | Disk 98% full | Old caches and downloads | Cleanup (5.9 → 30 GB free) |
| 2 | Setup | Can't train on MX230 | 2 GB, too old for current PyTorch | CPU-only torch locally; train on Colab |
| 3 | Data | `FileNotFoundError: data/train.jsonl` | `data/` folder missing; relative paths | `os.makedirs("data", exist_ok=True)`; always `cd` to the project |
| 4 | Data | 43% duplicates, leaked splits | Dataset merges two overlapping sources | Merge all splits, dedupe, re-split |
| 5 | Data | Broken commands pass `bash -n` | `bash -n` only checks bash grammar, not `find`'s rules | Extra rules: `-exec` terminator, truncation, quoted globs, macOS `sed` |
| 6 | Training | `torchao` ImportError | Colab's old torchao vs new PEFT | `pip uninstall -y torchao`, restart session, re-run from Cell 4 |
| 7 | Eval | Model outputs `'“man*”'` | Curly quotes in training data | `SMART_QUOTES` translate, retrain v2 |
| 8 | Eval | Base scores changed | Re-cleaning changed the test set | Compare models only on the same test set |
| 9 | Shell | `bin/env: No such file` + duplicate alias | Stale VS Code snap path in `.bashrc`; `>>` run twice | Comment out the line; delete the duplicate |
| 10 | howto | Answers non-shell questions with a URL | Narrow model can't refuse | Check the program exists before running |
| 11 | howto | `cd` does nothing | Child process can't change your shell | Tell the user to run it themselves |
| 12 | howto | `/bin/sh: ...` | `shell=True` uses dash | `executable="/bin/bash"` |
| 13 | Server | Port 8000 busy | Another app | Use 8765 |
| 14 | Docker | `flit_core` not found | Old pip in `python:3.10-slim` | `RUN pip install --upgrade pip` first |
| 15 | Docker | `FileNotFoundError` for a file that exists | `-rw-------` root-owned file, app runs as `appuser` | `useradd` before `COPY --chown=appuser:appuser` |
| 16 | Ollama | `--quantize` errors (twice) | Ollama 0.40 can't quantize GGUF; offers only MLX types | Quantize with llama.cpp `--outtype q8_0`, import without `--quantize` |
| 17 | howto | No warning for `rmdir` | `\brm\b` doesn't match `rmdir` | Pattern `\b(rm\|rmdir\|unlink\|shred)\b`; `test_howto.py` |
| 18 | GitHub | 403 denied to the work account | HTTPS remote used saved work login | SSH remote; separate key + `~/.ssh/config` host alias |
| 19 | AWS | IAM user steps don't exist | New sign-up flow account | Skip IAM; Builder ID + MFA is the everyday login |
| 20 | AWS | Can't pick a region / "upgrade" prompt | New flow assigns the region | Use the assigned region (Sydney); don't upgrade |
| 21 | Tunnel | Answer came from the laptop | Typo `SHELLSHELL_BUDDY_URL` | Spell the variable exactly; prove with `ollama stop` + `ollama ps` |
| 22 | howto | `MODEL_DIR` missing | Line lost during edits | Restored |
| 23 | Laptop | `python: command not found` | Plain `python` exists only inside a venv | `source .mb/bin/activate` or use `.mb/bin/python` / `python3` |

## Glossary

| Term | Meaning |
|---|---|
| **LLM** | Large language model: predicts the next token, repeatedly |
| **Token / tokenizer** | Text piece with a numeric ID / the lookup table that converts text ↔ IDs |
| **Chat template** | The exact text format (with `<\|im_start\|>` markers) a chat model was trained on |
| **Fine-tuning** | Extra training on your own examples for one task |
| **LoRA / adapter** | Train a small add-on (~35 MB) instead of the whole model |
| **Epoch** | One pass over all training data |
| **Loss** | How wrong the predictions are (lower is better) |
| **Overfitting** | Memorizing training data; validation loss rises |
| **Train / validation / test** | Learn from / monitor during training / final unseen exam |
| **Leakage** | Test examples also in training: inflated scores |
| **Merge** | Bake the adapter into the base model |
| **Quantization** | Rounding weights to fewer bits: smaller and faster |
| **GGUF** | Model file format used by llama.cpp and Ollama |
| **Ollama** | Program that runs GGUF models and serves an API (port 11434) |
| **API / endpoint** | A way for programs to talk to a service / one URL + method (e.g. `POST /suggest`) |
| **Port** | Numbered door on a computer; one program per port |
| **127.0.0.1 / 0.0.0.0** | "This computer only" / "all network interfaces" |
| **Environment variable** | Setting passed to a program from outside its code |
| **Docker image / container** | Frozen package / running instance of it |
| **systemd service** | Background program that restarts on crash and starts on boot |
| **EC2 instance** | A rented computer in AWS |
| **Region** | AWS data-center location (ours: Sydney) |
| **Security group** | The server's firewall |
| **Elastic IP** | A fixed public address |
| **SSH key pair** | Private key (never share) + public key (the lock) |
| **SSH tunnel** | Private pipe inside an SSH connection |
| **Reverse proxy** | Middleman that receives outside traffic and forwards it inside (Caddy) |
| **HTTPS / certificate** | Encrypted web traffic / the server's digital ID card |
| **API key** | A password for programs |
| **Rate limit** | Max requests per key per minute |
| **Revoke** | Cancel a key |

## Safety rules

- **Never run a command without reading it.** `howto` always asks; risky ones need the full word `yes`.
- **Never share private keys or API keys**: only `.pub` files go anywhere. Never put secrets in code or Git.
- **Check `git status` before every commit.** No `.mb/`, `data/`, model files or `.claude/`.
- **Keep repositories and registries private** unless you decide otherwise.
- **Read install scripts before running them with `sudo`.**
- **Don't expose a service** (`0.0.0.0`, open ports) without an API key, HTTPS and a rate limit. On office networks, ask IT first.
- **AWS:** budget alert first, MFA on, SSH from "My IP" only, open only the ports you need, stop the server when not in use, release the Elastic IP when done.
- **Don't paste real customer data or confidential work data** into external tools.

## What's next

- **v3: everyday commands.** 63% of the data is `find`, so `pwd`, `df -h`, `du`, `ps`, `tar` are weak. Add a few hundred varied examples → retrain on Colab → evaluate on the same test set → convert to `q8_0` → `ollama create shell-buddy:v3` (keep v2 to compare) → copy to the server.
- **Upgrade Python** from 3.10 (end-of-life) to 3.12+: rebuild `.mb` and re-test.
- **Execution-based evaluation:** run expected and predicted commands in a sandbox and compare outputs, instead of `exact_match`.
- **Publish the model** to Hugging Face Hub (handles large files), with a model card.
