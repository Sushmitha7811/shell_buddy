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
