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
