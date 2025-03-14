import torch
from datasets import Features, Value, load_dataset, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fuzzywuzzy import fuzz
import re
import os
from dotenv import load_dotenv
import logging
import time
from peft import get_peft_model, AdapterConfig, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer

logging.basicConfig(level=logging.INFO)
load_dotenv()

ACCESS_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
MODEL_DIR = os.getcwd() + "/models"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DATSET_DIR = os.getenv("DATASET_DIR") + "FinQA/dataset/"


# Modified training setup
def train():
    # Load model to train
    model, tokenizer = load_model(MODEL_ID, DEVICE, ACCESS_TOKEN, MODEL_DIR)
    train_dataset = load_preprocessed_dataset("json", data_files=DATSET_DIR+"/train.cleaned.json", split="train")


    number_of_samples = 22
    train_dataset = train_dataset.select(range(number_of_samples))

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            add_special_tokens=True
        )
    
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./adapters/checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_train_epochs=3,
        logging_dir="./adapters/logs",
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train only adapters
    trainer.train()
    
    # Save adapters
    model.save_pretrained("./adapters/fine_tuned_model")


# Modified load_model function with adapters
def load_model(model_id, device, token, cache_dir):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": device},
        quantization_config=quant_config,
        token=token,
        cache_dir=cache_dir
    )
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    # Add adapters (Houlsby style)
    adapter_config = AdapterConfig(
        peft_type="ADAPTER",
        adapter_type="houlsby",
        reduction_factor=16,  # Tune this
        mh_adapter=True,
        output_adapter=True,
        ln_after=False,
        ln_before=False,
    )
    
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=token,
        cache_dir=cache_dir
    )
    
    return model, tokenizer


def load_preprocessed_dataset(path: str, data_files: str, split: str):
    logging.info("Loading dataset")
    dataset = load_dataset(
        path,
        data_files=data_files,
        split=split
    )

    logging.info("Preprocessing dataset")
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    return dataset


def load_finetuned_model(model_id, device, token, cache_dir):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": device},
        quantization_config=quant_config,
        token=token,
        cache_dir=cache_dir
    )

    model = PeftModel.from_pretrained(model, "./adapters/fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=token,
        cache_dir=cache_dir
    )
    
    return model, tokenizer


def preprocess_function(example):
    # TODO: should raise error or warning when absent
    question = example["qa"].get("question", "No question available.")
    expected_answer = str(example['qa'].get("answer", "")).strip()
    table = example.get("table", [])
    table_str = "\n".join([" | ".join(row) for row in table])

    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))

    input_text = (
        "You are a financial calculator. Follow these steps:\n"
        "Return ONLY the final numerical or boolean answer with no text explanation\n\n"
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

# Add this to your main()
if __name__ == "__main__":
    train() 