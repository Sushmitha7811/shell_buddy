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
