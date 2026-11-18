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
