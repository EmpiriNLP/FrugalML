import os
import json
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fuzzywuzzy import fuzz
from datetime import datetime

# === CONFIG ===
MODEL_DIR = "/cs/student/projects2/aisd/2024/giliev/prefix_tuned_llama_3_1_8B_100tokens_3epochs_1batch_full"
TEST_FILE = "/cs/student/projects2/aisd/2024/giliev/FinQA/dataset/test.json"
OUTPUT_DIR = f"{MODEL_DIR}/test_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
THRESHOLD = 75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === HELPERS ===

def flatten(example):
    example["question"] = example["qa"].get("question", "")
    example["answer"] = example["qa"].get("answer", "")
    del example["qa"]
    return example

def preprocess(example):
    table = example.get("table", [])
    table_str = "\n".join([" | ".join(row) for row in table])
    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))
    question = example.get("question", "")
    expected_answer = str(example.get("answer", "")).strip()

    input_text = (
        "You are a financial calculator. Follow these steps:\n"
        "1. Extract relevant numerical information from the financial data.\n"
        "2. Calculate the answer to the question precisely.\n"
        "3. Return ONLY the final numerical answer with no explanation.\n"
        "4. Format percentages with % symbol (e.g. 15%).\n"
        "5. Round to the nearest whole number unless precision is required.\n\n"
        f"Example 1:\nQuestion: What was the revenue growth?\nFinal Answer (number only): 12%\n\n"
        f"Example 2:\nQuestion: What was the total assets value?\nFinal Answer (number only): 380\n\n"
        f"Pre Text Data:\n{pre_text}\n\n"
        f"Table Data:\n{table_str}\n\n"
        f"Post Text Data:\n{post_text}\n\n"
        f"Question: {question}\n"
        "Final Answer (number only): "
    )

    return {
        "input_text": input_text,
        "expected_answer": expected_answer,
        "question": question
    }

def clean_answer(text):
    percent_match = re.search(r'[-+]?\d*\.?\d+\s*%', text)
    if percent_match:
        number = float(percent_match.group(0).replace('%', '').strip())
        return f"{round(number, 0)}%"

    decimal_match = re.search(r'[-+]?\d*\.?\d+', text)
    if decimal_match:
        number = float(decimal_match.group(0))
        if abs(round(number) - number) < 0.01:
            return str(round(number))
        return str(round(number, 0))

    text = text.lower().strip()
    if 'yes' in text or 'true' in text:
        return 'yes'
    if 'no' in text or 'false' in text:
        return 'no'

    return text.strip()

def generate_answer(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.05,
            top_k=5,
            max_new_tokens=10,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

# === MAIN ===

# Load test data
with open(TEST_FILE, "r") as f:
    raw_test_data = json.load(f)

test_data = [flatten(e) for e in raw_test_data]
test_dataset = Dataset.from_list(test_data).map(preprocess)

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token  # ← Add this to suppress generation warnings
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto", use_cache=False)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

# Evaluation loop
results = []
correct = 0
similarities = []

print("Starting evaluation...")
for i, example in enumerate(test_dataset):
    pred = generate_answer(example["input_text"], model, tokenizer)
    pred_clean = clean_answer(pred)
    target_clean = clean_answer(example["expected_answer"])
    similarity = fuzz.ratio(pred_clean.lower(), target_clean.lower())
    is_correct = similarity >= THRESHOLD

    results.append({
        "question": example["question"],
        "expected": example["expected_answer"],
        "predicted": pred,
        "cleaned_pred": pred_clean,
        "cleaned_expected": target_clean,
        "similarity": similarity,
        "correct": is_correct
    })

    if is_correct:
        correct += 1

    if i < 10:
        print(f"\n[Example {i+1}]")
        print(f"Question: {example['question']}")
        print(f"Expected: {example['expected_answer']} → {target_clean}")
        print(f"Predicted: {pred} → {pred_clean}")
        print(f"Similarity: {similarity}")

# Metrics
accuracy = correct / len(results) * 100
avg_similarity = sum(r["similarity"] for r in results) / len(results)

summary = {
    "model": MODEL_DIR,
    "test_file": TEST_FILE,
    "total_samples": len(results),
    "accuracy": f"{accuracy:.2f}%",
    "avg_similarity": f"{avg_similarity:.2f}%"
}

# Save results
with open(f"{OUTPUT_DIR}/test_predictions.json", "w") as f:
    json.dump(results, f, indent=2)

with open(f"{OUTPUT_DIR}/test_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Test Set Evaluation Complete ===")
print(json.dumps(summary, indent=2))
