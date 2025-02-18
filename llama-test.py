from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
import time
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

MODELS_DIRECTORY = "D:/models/huggingface/"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_DIRECTORY = "D:/datasets/EmpiriNLP/FinQA/dataset"

num_outputs = 1

messages = [
    {"role": "system", "content": "You are a financial expert assistant that produces numerical predictions upon reasoning context."},
    {"role": "user", "content": "Who are you?"},
]

def load_model(model_id, models_directory):
    tokeniser = AutoTokenizer.from_pretrained(os.path.join(models_directory, model_id))
    model = AutoModelForCausalLM.from_pretrained(os.path.join(models_directory, model_id), torch_dtype=torch.bfloat16, device_map="auto")
    return model, tokeniser

def load_dataset(dataset_directory: str, split: str = "train"):
    tsv_data = pd.read_csv(os.path.join(dataset_directory, f"{split}.tsv"), sep="\t")
    return tsv_data

def extract_qa_pairs(dataset: pd.DataFrame, number_of_pairs: int = 5):
    q_cols = ["pre_text", "table", "post_text", "question"]
    data = dataset.loc[:number_of_pairs, q_cols + ["answer"]].copy()
    data["question"] = data[q_cols].apply(lambda x: "\n".join([str(x[q]) for q in q_cols]), axis=1)
    return data[["question", "answer"]]

def create_message_from_qa_pair(qa_pair: pd.Series, tokeniser: AutoTokenizer=None, chat_template: bool = False):
    if not chat_template:
        return qa_pair["question"]
    else:
        template = [
            {"role": "system", "content": "You are a financial expert assistant that uses numerical reasoning to answer questions from snippets of text and tables. Provide strictly only the answer in one number or word."},
            {"role": "user", "content": qa_pair["question"]}
        ]

        return tokeniser.apply_chat_template(template, tokenize=False, add_generation_prompt=True)

def create_message_input(dataset: pd.DataFrame, number_of_messages: int = 5, tokeniser: AutoTokenizer = None, chat_template: bool = False):
    data = extract_qa_pairs(dataset, number_of_messages)
    for i in range(number_of_messages):
        yield create_message_from_qa_pair(data.iloc[i], tokeniser=tokeniser, chat_template=chat_template)
    

if __name__ == "__main__":

    start_time = time.time()
    pipe = pipeline("text-generation", model=MODEL_ID, device_map="auto", model_kwargs={"torch_dtype": torch.bfloat16})
    tokeniser = pipe.tokenizer
    tokeniser.pad_token_id = tokeniser.eos_token_id # Fix padding token
    logging.info(f"Pipeline creation in {time.time() - start_time:.2f} seconds" + "-"*50)

    EXAMPLE_SIZE = 24
    BATCH_SIZE = 8
    dataset = load_dataset(DATASET_DIRECTORY)

    start_time = time.time()
    outputs = []
    for output in pipe(create_message_input(dataset, EXAMPLE_SIZE, tokeniser=tokeniser, chat_template=True), max_new_tokens=20, num_return_sequences=num_outputs, return_full_text=False):
        outputs.append(output[0]["generated_text"])
    logging.debug(f"Iterative generation in {time.time() - start_time:.2f} seconds" + "-"*50)

    for i, output in enumerate(outputs):
        print(f"Question: {dataset.iloc[i]['question']}")
        print(f"Correct Answer: {dataset.iloc[i]['answer']}")
        print(f"Model Answer: {output}")
        print("-"*50)


    start_time = time.time()
    outputs = []
    for output in pipe(create_message_input(dataset, EXAMPLE_SIZE, tokeniser=tokeniser, chat_template=True), max_new_tokens=10, num_return_sequences=num_outputs, return_full_text=False, batch_size=BATCH_SIZE):
        outputs.append(output[0]["generated_text"])
    logging.debug(f"Batched generation in {time.time() - start_time:.2f} seconds" + "-"*50)

    for i, output in enumerate(outputs):
        print(f"Question: {dataset.iloc[i]['question']}")
        print(f"Correct Answer: {dataset.iloc[i]['answer']}")
        print(f"Model Answer: {output}")
        print("-"*50)

    # answer = outputs[0]["generated_text"]
    # print(answer)
