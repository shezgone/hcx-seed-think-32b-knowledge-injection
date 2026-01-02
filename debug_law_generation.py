import mlx.core as mx
from mlx_lm import load, generate
from datasets import load_dataset

model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
print(f"Loading model from {model_path}...")
model, tokenizer = load(model_path)

print("Loading Law dataset...")
dataset = load_dataset("HAERAE-HUB/KMMLU", "Law", split="test")
example = dataset[0]
print(f"Example keys: {example.keys()}")

question = example["question"]
if "options" in example:
    options = example["options"]
elif "A" in example:
    options = [example["A"], example["B"], example["C"], example["D"]]
else:
    options = ["Option 1", "Option 2", "Option 3", "Option 4"] # Fallback

choices = ["A", "B", "C", "D"]

prompt = f"<|im_start|>user\n다음 문제를 읽고 정답을 하나만 고르시오.\n\n질문: {question}\n"
for j, opt in enumerate(options):
    prompt += f"{choices[j]}. {opt}\n"
prompt += "\n정답:<|im_end|>\n<|im_start|>assistant\n정답은"

print("\n=== Prompt ===")
print(prompt)

print("\n=== Generation ===")
response = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=True)
print(f"\nFull Response: '{response}'")
