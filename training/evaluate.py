import json
import re
import subprocess
import sys
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "shell-buddy-lora-v2"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100   # how many test rows to score

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.float32)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
base.eval()

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
