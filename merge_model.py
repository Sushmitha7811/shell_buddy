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
