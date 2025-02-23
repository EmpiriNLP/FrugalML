from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import logging
import time
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

MODELS_DIRECTORY = "D:/models/huggingface/"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_DIRECTORY = "D:/datasets/EmpiriNLP/FinQA/dataset"

def log_time(level, process_name: str = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if level == "debug":
                logging.debug(f"Time taken for {process_name}: {end - start:.2f}s")
            elif level == "info":
                logging.info(f"Time taken for {process_name}: {end - start:.2f}s")
            else:
                logging.warning(f"Time taken for {process_name}: {end - start:.2f}s")
            return result
        return wrapper
    return decorator

@log_time("info", process_name="Load Model")
def initialise_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # quantization_config=quantization_config
    )

    return model, tokenizer

@log_time("info", process_name="Load Dataset")
def load_dataset(dataset_directory: str, split: str = "train"):
    tsv_data = pd.read_csv(os.path.join(dataset_directory, f"{split}.tsv"), sep="\t")
    return tsv_data

@log_time("info", process_name="Preprocess Context")
def preprocess_context(context: pd.Series, tokeniser: AutoTokenizer):

    q_cols = ["pre_text", "table", "post_text", "question"]
    data = context[q_cols].copy()

    template = [
        {"role": "system", "content": "You are a financial expert assistant that uses numerical reasoning to answer questions from snippets of text and tables. Provide strictly only the answer in one number or word."},
        {"role": "user", "content": "\n".join(data.values)}
    ]

    tokeniser.pad_token = tokeniser.eos_token

    input_ids = tokeniser.apply_chat_template(template, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_attention_mask=True, truncation=True, padding="max_length", max_length=2048)
    
    return input_ids

@log_time("debug", process_name="Generate Message")
def generate_answer(inputs: list[int], model: AutoModelForCausalLM, tokeniser: AutoTokenizer):
    # inputs = torch.tensor(inputs).unsqueeze(0).to(model.device) # input ids and attention mask
    output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokeniser.eos_token_id, eos_token_id=tokeniser.eos_token_id)

    return tokeniser.decode(output[0], skip_special_tokens=True)

@log_time("debug", process_name="Inference")
def inference(train_data: pd.DataFrame, model: AutoModelForCausalLM, tokeniser: AutoTokenizer, number_of_pairs: int = 5):
    for i in range(number_of_pairs):
        question = train_data.iloc[i]["question"]
        answer = train_data.iloc[i]["answer"]

        inputs = preprocess_context(train_data.iloc[i], tokeniser)
        generated_answer = generate_answer(inputs, model, tokeniser)

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Generated Answer: {generated_answer}")
        print("-"*50)

def main():
    model, tokeniser = initialise_model(MODEL_ID)
    print("Model initialised successfully!")
    
    # Load datasets
    train_data = load_dataset(DATASET_DIRECTORY, split="train")
    print("Dataset loaded successfully!")
    
    # Evaluate model
    inference(train_data, model, tokeniser, number_of_pairs=5)

if __name__ == "__main__":
    main()