---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
- AnishJoshi/nl2bash-custom
language:
- en
tags:
- gguf
- bash
- shell
- ollama
pipeline_tag: text-generation
---

# Shell Buddy (GGUF)

A small model that turns plain English into a single Linux bash command.
Fine-tuned from Qwen2.5-0.5B-Instruct and quantized to 8-bit (531 MB), so it runs on an ordinary laptop CPU.

## Try it

    ollama run hf.co/SushmithaJujare/shell-buddy-GGUF "find all files bigger than 1GB in my home folder"

## Results (100 unseen examples)

| Model | Right tool chosen | Exact match |
|---|---|---|
| Base Qwen 0.5B (not fine-tuned) | 62% | 5% |
| Qwen 7B (14× bigger, not fine-tuned) | 63% | 9% |
| **Shell Buddy (0.5B, fine-tuned, 8-bit)** | **91%** | **18%** |

## How it was made

- ~10,800 English → bash pairs cleaned from NL2Bash
- LoRA fine-tuning on a free Google Colab GPU
- Merged and converted to GGUF (q8_0) with llama.cpp

## Limitations

- It can be wrong. **Always read a command before running it.**
- Weaker on everyday commands like `pwd`, `df -h` and `du`, because most of the training data uses `find`.
