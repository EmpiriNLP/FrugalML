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
def preprocess_context(context: pd.Series, tokeniser: AutoTokenizer, chat_template: bool = False):

    relevant_info = context["gold_inds"]
    table_str = context["table"]
    question = context["question"]
    program = context["program"]

    # Broken
    if chat_template:
        template = [
            {"role": "system", "content": "You are a financial expert assistant that uses numerical reasoning to answer questions from snippets of text and tables. Provide strictly only the answer in one number or word."},
            {"role": "user", "content": "\n".join(data.values)}
        ]
        tokeniser.pad_token = tokeniser.eos_token
        inputs = tokeniser.apply_chat_template(template, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_attention_mask=True, truncation=True, padding="max_length", max_length=2048)

        return inputs

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

    tokeniser.pad_token = tokeniser.eos_token
    inputs = tokeniser(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=2048)

    return inputs

@log_time("debug", process_name="Generate Message")
def generate_answer(inputs: list[int], model: AutoModelForCausalLM, tokeniser: AutoTokenizer):
    inputs = {k: v.to(model.device) for k, v in inputs.items()} 
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