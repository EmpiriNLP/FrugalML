import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def initialize_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    return model, tokenizer

def load_finqa_dataset(path):
    with open(path, "r") as f:
        return json.load(f)

def simple_inference(example, model, tokenizer):
    """Generate numerical answers with minimal context and clear instruction"""
    question = example["qa"]["question"]
    
    # Extract only the most relevant information from FinQA
    relevant_info = " ".join(example["qa"].get("gold_inds", {}).values())

    # Ensure the model knows it's supposed to return ONLY a number
    input_text = (
        "You are a financial calculator. "
        "Read the question carefully and return ONLY the final numerical answer without explanation.\n\n"
        f"Relevant Information:\n{relevant_info}\n\n"
        f"Question: {question}\n"
        "Final Answer (number only): "
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            num_beams=1,  # No beam search
            do_sample=False  # Ensure deterministic output
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

def evaluate_baseline(test_data, model, tokenizer, num_samples=10):
    """Simple evaluation loop"""
    for i, example in enumerate(test_data[:num_samples]):
        try:
            prediction = simple_inference(example, model, tokenizer)
            ground_truth = example["qa"]["answer"]
            
            print(f"\nExample {i+1}")
            print(f"Question: {example['qa']['question']}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Raw Prediction: {prediction}")
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")

def main():
    model, tokenizer = initialize_model()
    print("Model initialized successfully!")
    
    test_data = load_finqa_dataset("/cs/student/projects2/aisd/2024/giliev/FinQA/dataset/test.json")
    print("Dataset loaded successfully!")
    
    evaluate_baseline(test_data, model, tokenizer)

if __name__ == "__main__":
    main()