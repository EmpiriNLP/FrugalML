import os
import time
import json
import re
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DefaultDataCollator
)
from peft import get_peft_model, PrefixTuningConfig, TaskType
from fuzzywuzzy import fuzz
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prefix_tuning_experiment2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_FILE = "/cs/student/projects2/aisd/2024/giliev/FinQA/dataset/train.json"
    DEV_FILE = "/cs/student/projects2/aisd/2024/giliev/FinQA/dataset/dev.json"

    PREFIX_TOKENS = 101
    EPOCHS = 5
    BATCH_SIZE = 1
    NUM_TRAIN_SAMPLES = None
    NUM_EVAL_SAMPLES = None

    OUTPUT_DIR = f"./secondprefix_tuned_llama_3_1_8B_{PREFIX_TOKENS}tokens_{EPOCHS}epochs_{BATCH_SIZE}batch_full"
    THRESHOLD = 75

# Load and preprocess data
def preprocess_function(example):
    question = example["qa"].get("question", "No question available.")
    expected_answer = str(example['qa'].get("answer", "")).strip()
    table = example.get("table", [])
    table_str = "\n".join([" | ".join(row) for row in table])
    
    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))

    # Enhanced prompt with clearer instructions and examples
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
        "expected_answer": expected_answer
    }

class FinancialDataset(TorchDataset):
    def __init__(self, examples, tokenizer, max_length=1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples) 
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["input_text"]
        expected_answer = example["expected_answer"]
        
        # Use fixed sequence length for all examples
        fixed_input_length = self.max_length - 32  # Reserve space for answer
        
        # Tokenize input prompt with padding
        tokenized_prompt = self.tokenizer(
            input_text, 
            truncation=True,
            max_length=fixed_input_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize expected completion
        tokenized_answer = self.tokenizer(
            expected_answer,
            truncation=True,
            max_length=32,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create input_ids and attention_mask
        input_ids = tokenized_prompt["input_ids"].squeeze(0)
        answer_ids = tokenized_answer["input_ids"].squeeze(0)
        
        # Create labels (-100 for prompt, actual IDs for answer)
        labels = torch.clone(input_ids)
        labels[:] = -100  # Mark all input tokens to be ignored in loss
        
        # Concatenate for full sequence
        full_input_ids = torch.cat([input_ids, answer_ids])
        full_attention_mask = torch.cat([
            tokenized_prompt["attention_mask"].squeeze(0),
            tokenized_answer["attention_mask"].squeeze(0)
        ])
        
        # Extend labels with answer tokens
        full_labels = torch.cat([
            labels,
            answer_ids
        ])
        
        # Ensure all tensors have consistent length
        max_len = min(self.max_length, len(full_input_ids))
        full_input_ids = full_input_ids[:max_len]
        full_attention_mask = full_attention_mask[:max_len]
        full_labels = full_labels[:max_len]
            
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": full_labels
        }

def load_and_prepare_data(config):
    logger.info("Loading FinQA train and dev datasets")

    # Load raw JSON
    with open(config.TRAIN_FILE, "r") as f:
        train_data = json.load(f)
    with open(config.DEV_FILE, "r") as f:
        eval_data = json.load(f)

    # Flatten 'qa' field to top level
    def flatten(example):
        example["question"] = example["qa"].get("question", "")
        example["answer"] = example["qa"].get("answer", "")
        del example["qa"]
        return example

    train_data = [flatten(e) for e in train_data]
    eval_data = [flatten(e) for e in eval_data]

    # Convert to HF Datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    logger.info(f"Original Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

    # Update the preprocessing function to use the flattened structure
    def preprocess_flat(example):
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
            "expected_answer": expected_answer
        }

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_flat)
    eval_dataset = eval_dataset.map(preprocess_flat)
    
    config.NUM_TRAIN_SAMPLES = len(train_dataset)
    config.NUM_EVAL_SAMPLES = len(eval_dataset)

    return train_dataset, eval_dataset

def initialize_model_for_prefix_tuning(config):
    logger.info(f"Initializing model: {config.MODEL_ID}")
    start_time = time.time()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in half precision without 4-bit quantization for proper gradient flow
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float16,  # Use float16 instead of 4-bit to allow gradient computation
        device_map="auto",
        use_cache=False  # Critical for training with gradient computation
    )

    # Configure Prefix Tuning
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=config.PREFIX_TOKENS,
        prefix_projection=True,
        inference_mode=False,  # Must be False for training
        token_dim=model.config.hidden_size,
        num_layers=model.config.num_hidden_layers,
        num_attention_heads=model.config.num_attention_heads
    )
    
    # Apply Prefix Tuning
    peft_model = get_peft_model(model, peft_config)
    
    # Log trainable parameters info
    trainable_params = 0
    all_params = 0
    for _, param in peft_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of all params)")
    logger.info(f"Model initialization time: {time.time() - start_time:.2f}s")
    
    return peft_model, tokenizer

def train_prefix_tuning(config, model, tokenizer, train_dataset_raw, eval_dataset_raw):
    logger.info("Starting prefix tuning training")
    start_time = time.time()
    
    # Create PyTorch datasets
    train_dataset = FinancialDataset(train_dataset_raw, tokenizer)
    eval_dataset = FinancialDataset(eval_dataset_raw, tokenizer)

    # Define training arguments
    # In the train_prefix_tuning function:
    # In the train_prefix_tuning function, update the TrainingArguments:
    training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    fp16=True,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir=f"{config.OUTPUT_DIR}/logs",
    logging_steps=5,          # less frequent logs
    eval_strategy="epoch",    # evaluate less frequently (per epoch)
    save_strategy="epoch",    # save only once per epoch
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    report_to="none",
    )

    # Create data collator that will handle padding consistently
    data_collator = DefaultDataCollator()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    model.save_pretrained(config.OUTPUT_DIR)
    logger.info(f"Model saved to {config.OUTPUT_DIR}")
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f}s")
    
    return model, training_time

def clean_answer(text):
    """Extract and format numerical answer"""
    # Handle percentages first
    percent_match = re.search(r'[-+]?\d*\.?\d+\s*%', text)
    if percent_match:
        number = float(percent_match.group(0).replace('%', '').strip())
        return f"{round(number, 0)}%"

    # Handle regular numbers
    decimal_match = re.search(r'[-+]?\d*\.?\d+', text, re.MULTILINE)
    if decimal_match:
        number = float(decimal_match.group(0))
        if abs(round(number) - number) < 0.01:
            return str(round(number))
        return str(round(number, 0))

    # Handle yes/no answers
    text = text.lower().strip()
    if 'yes' in text or 'true' in text:
        return 'yes'
    if 'no' in text or 'false' in text:
        return 'no'

    return text.strip()

def generate_answer(input_text, model, tokenizer, device):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.05,
            top_k=5,
            max_new_tokens=20,
            do_sample=False,
            num_beams=5,
            early_stopping=True
        )
    # Add early stopping to the training arguments
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
    
    generated_text = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return generated_text.strip()

def evaluate_model(config, model, tokenizer, train_dataset_raw, eval_dataset_raw):
    logger.info("Starting model evaluation")
    start_time = time.time()
    
    total_samples = len(eval_dataset_raw)
    correct_predictions = 0
    similarity_scores = []
    
    for i, example in enumerate(eval_dataset_raw):
        predicted_answer = generate_answer(example["input_text"], model, tokenizer, config.DEVICE)
        expected_answer = example["expected_answer"]
        
        clean_p = clean_answer(predicted_answer)
        clean_e = clean_answer(expected_answer)
        
        similarity = fuzz.ratio(clean_p.lower(), clean_e.lower())
        similarity_scores.append(similarity)
        
        if i < 10:  # Print first 10 examples for inspection
            logger.info(f"\n*** Example {i+1} ***")
            logger.info(f"Question: {example['input_text'].split('Question:')[1].split('Final Answer')[0]}")
            logger.info(f"Expected: {expected_answer} (cleaned: {clean_e})")
            logger.info(f"Predicted: {predicted_answer} (cleaned: {clean_p})")
            logger.info(f"Similarity: {similarity}")
        
        # Count as correct if similarity is above threshold
        if similarity >= config.THRESHOLD:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_samples * 100
    avg_similarity = sum(similarity_scores) / total_samples if similarity_scores else 0
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / total_samples if total_samples else 0
    
    results = {
        "model": config.MODEL_ID,
        "experiment": f"Prefix Tuning ({config.PREFIX_TOKENS} tokens, {config.EPOCHS} epochs, batch={config.BATCH_SIZE})",
        "trained_samples": len(train_dataset_raw),
        "evaluated_samples": len(eval_dataset_raw),
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "evaluated_samples": total_samples,
        "accuracy": f"{accuracy:.2f}%",
        "avg_similarity": f"{avg_similarity:.2f}%",
        "total_time": f"{total_time:.2f}s",
        "avg_time_per_sample": f"{avg_time_per_sample:.2f}s",
    }
    
    logger.info("Evaluation Results:")
    for key, value in results.items():
        logger.info(f"{key}: {value}")
    
    # Save results to file
    with open(f"{config.OUTPUT_DIR}/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    config = Config()
    logger.info(f"Starting experiment with: {config.PREFIX_TOKENS} tokens, {config.EPOCHS} epochs, batch size {config.BATCH_SIZE}, {config.NUM_TRAIN_SAMPLES} training samples")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    train_dataset_raw, eval_dataset_raw = load_and_prepare_data(config)
    
    # Initialize model
    model, tokenizer = initialize_model_for_prefix_tuning(config)
    
    # Record start time for total experiment
    exp_start_time = time.time()
    
    # Train model
    model, training_time = train_prefix_tuning(config, model, tokenizer, train_dataset_raw, eval_dataset_raw)
    
    # Evaluate model
    results = evaluate_model(config, model, tokenizer, train_dataset_raw, eval_dataset_raw)
    
    # Update results with training time
    results["training_time"] = f"{training_time:.2f}s"
    results["total_experiment_time"] = f"{time.time() - exp_start_time:.2f}s"
    
    # Save final results
    with open(f"{config.OUTPUT_DIR}/final_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Experiment completed. Results saved to {config.OUTPUT_DIR}/final_results.json")
    
    return results

if __name__ == "__main__":
    main()