from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
import time

logging.basicConfig(level=logging.INFO)

MODELS_DIRECTORY = "D:/models/huggingface/"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_DIRECTORY = "D:/datasets/EmpiriNLP/FinQA/dataset"

number_of_return_sequences = 1

messages = [
    {"role": "system", "content": "You are a financial expert assistant that produces numerical predictions upon reasoning context."},
    {"role": "user", "content": "Who are you?"},
]

def load_model(model_id, models_directory):
    tokeniser = AutoTokenizer.from_pretrained(os.path.join(models_directory, model_id))
    model = AutoModelForCausalLM.from_pretrained(os.path.join(models_directory, model_id), torch_dtype=torch.bfloat16, device_map="auto")
    return model, tokeniser


if __name__ == "__main__":

    start_time = time.time()
    model, tokeniser = load_model(MODEL_ID, MODELS_DIRECTORY)
    pipe = pipeline("text-generation", model=MODEL_ID, device_map="auto")
    logging.info(f"Pipeline loaded in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    outputs = pipe(messages, max_new_tokens=100, num_return_sequences=number_of_return_sequences, return_full_text=False)
    logging.info(f"Generated in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    answer = outputs[0]["generated_text"]
    logging.info(f"Answer extracted in {time.time() - start_time:.2f} seconds")

    print(answer)