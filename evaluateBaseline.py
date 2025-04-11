# evaluate_baseline.py
import os
import json
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from fuzzywuzzy import fuzz
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
TEST_FILE = "/cs/student/projects2/aisd/2024/giliev/FinQA/dataset/test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 75
MAX_NEW_TOKENS = 20

# Preprocessing
def preprocess(example):
    table = example.get("table", [])
    table_str = "\n".join([" | ".join(row) for row in table])
    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))
    question = example.get("qa", {}).get("question", "")
    answer = str(example.get("qa", {}).get("answer", "")).strip()

    input_text = (
        "You are a financial calculator. Follow these steps:\n"
        "1. Extract relevant numerical information from the financial data.\n"
        "2. Calculate the answer to the question precisely.\n"
        "3. Return ONLY the final numerical answer with no explanation.\n"
        "4. Format percentages with % symbol (e.g. 15%).\n"
        "5. Round to the nearest whole number unless precision is required.\n\n"
        f"Pre Text Data:\n{pre_text}\n\n"
        f"Table Data:\n{table_str}\n\n"
        f"Post Text Data:\n{post_text}\n\n"
        f"Question: {question}\n"
        "Final Answer (number only): "
    )
    return {
        "input_text": input_text,
        "expected_answer": answer
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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return generated.strip()

def main():
    logger.info("Loading test data...")
    with open(TEST_FILE) as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list(raw_data).map(preprocess)

    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

    correct = 0
    similarity_scores = []

    logger.info("Starting evaluation...")
    for i, example in enumerate(dataset):
        pred = generate_answer(example["input_text"], model, tokenizer)
        expected = example["expected_answer"]

        pred_clean = clean_answer(pred)
        expected_clean = clean_answer(expected)
        sim = fuzz.ratio(pred_clean.lower(), expected_clean.lower())
        similarity_scores.append(sim)

        if sim >= THRESHOLD:
            correct += 1

        if i < 10:
            logger.info(f"\n[Example {i+1}]\nQuestion: {example['input_text'].split('Question:')[-1]}\nExpected: {expected} → {expected_clean}\nPredicted: {pred} → {pred_clean}\nSimilarity: {sim}")

    accuracy = 100 * correct / len(dataset)
    avg_similarity = sum(similarity_scores) / len(dataset)

    logger.info(f"\n=== Baseline Evaluation Complete ===\nTotal Samples: {len(dataset)}\nAccuracy: {accuracy:.2f}%\nAvg Similarity: {avg_similarity:.2f}%")

if __name__ == "__main__":
    main()
