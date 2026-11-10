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


# def suggest_via_api(request):
#     """Ask a running Shell Buddy server. Returns None if no server is reachable."""
#     body = json.dumps({"request": request}).encode()
#     req = urllib.request.Request(f"{API_URL}/suggest", data=body, headers={"Content-Type": "application/json"})
#     try:
#         with urllib.request.urlopen(req, timeout=30) as resp:
#             return json.loads(resp.read())["command"]
#     except (urllib.error.URLError, TimeoutError):
#         return None

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
