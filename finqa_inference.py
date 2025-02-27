import json
import re
import torch
import pandas as pd
import numpy as np
from datasets import load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from huggingface_hub import login

# Model initialization
def initialize_model():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "microsoft/Phi-3-mini-4k-instruct"
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize with 4-bit quantization for better memory efficiency
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        # quantization_config=quantization_config
    )
    return model, tokenizer

# Data loading
def load_finqa_dataset(path):
    with open(path, "r") as f:
        return json.load(f)

def preprocess_example(example, tokenizer):
    """Prepare structured input for Phi-3"""
    question = example["qa"].get("question", "No question available.")
    table = example.get("table", [])
    table_str = "\n".join([" | ".join(row) for row in table])
    
    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))
    
    gold_inds = example["qa"].get("gold_inds", {})
    relevant_info = "\n".join(gold_inds.values())
    program = example["qa"].get("program", "")
    
    input_text = (
        "You are a financial calculator. Follow these steps:\n"
        "1. Read the question carefully\n"
        "2. Look at the relevant information and table data\n"
        "3. Follow the mathematical operation exactly\n"
        "4. Return ONLY the final numerical answer with no text\n\n"
        f"Relevant Information:\n{relevant_info}\n\n"
        f"Table Data:\n{table_str}\n\n"
        f"Question: {question}\n"
        f"Mathematical Operation: {program}\n"
        "Final Answer (number only): "
    )

    return tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt"
    )

def clean_answer(text):
    """Extract and format numerical answer"""
    if ':' in text:
        text = text.split(':')[-1]
    
    # Handle percentages first
    percent_match = re.search(r'[-+]?\d*\.?\d+\s*%?', text)
    if percent_match:
        number = float(percent_match.group(0).replace('%', '').strip())
        # If the number is small (likely decimal), convert to percentage
        if number < 1:
            number *= 100
        # Round to one decimal place and add % symbol
        return f"{round(number, 1)}%"
    
    # Handle regular numbers
    decimal_match = re.search(r'[-+]?\d*\.?\d+', text)
    if decimal_match:
        number = float(decimal_match.group(0))
        # If it's close to an integer, round it
        if abs(round(number) - number) < 0.01:
            return str(round(number))
        # Otherwise, round to one decimal place
        return str(round(number, 1))
    
    # Handle yes/no answers
    text = text.lower().strip()
    if 'yes' in text or 'true' in text:
        return 'yes'
    if 'no' in text or 'false' in text:
        return 'no'
    
    return text.strip()

def generate_answer(example, model, tokenizer):
    """Generate answer using Phi-3"""
    inputs = preprocess_example(example, tokenizer)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            num_beams=5,
            temperature=0.1,
            top_p=0.95,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_answer(generated_text)

def evaluate_model(test_data, model, tokenizer, num_samples=10):
    """Evaluate model performance"""
    metric = load_metric("squad", trust_remote_code=True)
    predictions = []
    references = []
    
    torch.cuda.empty_cache()
    
    for i, example in enumerate(test_data[:num_samples]):
        try:
            start_time = time.time()
            pred_text = generate_answer(example, model, tokenizer)
            true_text = example["qa"]["answer"]
            
            predictions.append({"id": str(i), "prediction_text": pred_text})
            references.append({"id": str(i), "answers": {"text": [true_text], "answer_start": [0]}})
            
            em = 1 if pred_text.strip() == true_text.strip() else 0
            f1 = metric.compute(
                predictions=[{"id": str(i), "prediction_text": pred_text}],
                references=[{"id": str(i), "answers": {"text": [true_text], "answer_start": [0]}}]
            )["f1"]
            
            print(f"\n🔹 Example {i+1}")
            print(f"❓ Question: {example['qa']['question']}")
            print(f"✅ Ground Truth: {true_text}")
            print(f"🤖 Prediction: {pred_text}")
            print(f"⏱️ Time taken: {time.time() - start_time}")
            print(f"📊 Metrics - Exact Match: {em}, F1: {f1:.2f}")
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            continue
    
    results = metric.compute(predictions=predictions, references=references)
    print("\n📊 Overall Results:", results)
    return results
    # return None

def main():
    # Initialize model and tokenizer
    model, tokenizer = initialize_model()
    tokenizer.pad_token = tokenizer.eos_token
    print("Model initialized successfully!")
    
    # Load datasets
    test_data = load_finqa_dataset("D:/datasets/EmpiriNLP/FinQA/dataset/test.json")
    print("Dataset loaded successfully!")
    
    # Evaluate model
    results = evaluate_model(test_data, model, tokenizer, num_samples=8)
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()