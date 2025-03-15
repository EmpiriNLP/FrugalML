import torch
from datasets import Features, Value, load_dataset, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, EvalPrediction
from fuzzywuzzy import fuzz
import re
import os
from dotenv import load_dotenv
import logging
import time
from adapters import AutoAdapterModel, AdapterTrainer


logging.basicConfig(level=logging.INFO)
load_dotenv()

ACCESS_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
MODEL_DIR = os.getcwd() + "/models"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
DATSET_DIR = os.getenv("DATASET_DIR") + "FinQA/dataset/"
SIMILARITY_THRESHOLD = 85  



def main():
    model, tokenizer = load_model(MODEL_ID, DEVICE, ACCESS_TOKEN, MODEL_DIR)
    train_dataset = load_preprocessed_dataset("json", data_files=DATSET_DIR+"/train.cleaned.json", split="train")

    
    def preprocess_to_finetune(batch):

        encoded = tokenizer(batch["input_text"], max_length=80, truncation=True, padding="max_length")
        return  {key: torch.tensor(value).to(DEVICE) for key, value in encoded.items()}



    train_dataset = train_dataset.map(preprocess_to_finetune, batched=True)
    train_dataset = train_dataset.rename_column(original_column_name="expected_answer", new_column_name="labels")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) 
    train_dataset = train_dataset.with_format("torch", device=DEVICE)


    number_of_samples = 1
    eval_dataset = train_dataset.select(range(22))
    train_dataset = train_dataset.select(range(number_of_samples))

    model.add_adapter("finance_adapter", config="seq_bn")
    model.train_adapter("finance_adapter")

    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        output_dir="./models/adapter",
        overwrite_output_dir=True,
        # evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        # save_steps=10,
        # disable_tqdm=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    # Inference as before
    train_dataset = load_preprocessed_dataset("json", data_files=DATSET_DIR+"/train.cleaned.json", split="train")


    number_of_samples = 22
    train_dataset = train_dataset.select(range(number_of_samples))

    correct_predictions = 0
    threshold = 85  
    similarity_scores = []

    logging.info("Starting Inference")
    start = time.time()
    for i, example in enumerate(train_dataset):

        predicted_answer = generate_answer(example["input_text"], tokenizer, model)
        expected_answer = example["expected_answer"]

        clean_p = clean_answer(predicted_answer)
        clean_e = clean_answer(expected_answer)

        similarity = fuzz.ratio(clean_p.lower(), clean_e.lower())
        similarity_scores.append(similarity)

        if similarity >= threshold:
            correct_predictions += 1

        # Log every 10% of the dataset
        if number_of_samples >=100 and (i + 1) % (number_of_samples // 10) == 0:
            logging.info(f"Processed {i + 1} / {(i+1)/number_of_samples*100:.2f}%")

    end = time.time()

    accuracy = correct_predictions / number_of_samples * 100
    avg_similarity = sum(similarity_scores) / number_of_samples
    elapsed_time = end - start
    avg_elapsed_time = elapsed_time / number_of_samples

    logging.info(f"Total Samples: {number_of_samples}")
    logging.info(f"Correct Predictions: {correct_predictions}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"Average Similarity Score: {avg_similarity:.2f}%")


    logging.info(f"Total Inf time: {elapsed_time:.4f} seconds")
    logging.info(f"Avg Inf time: {avg_elapsed_time:.4f} seconds")



def load_model(model_id, device, token, cache_dir):
    logging.info(f"Loading model {model_id} on device {device}")
    # Define 4-bit quantization configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation
        bnb_4bit_use_double_quant=True,  # Double quantization for better memory efficiency
        bnb_4bit_quant_type="nf4"  # NormalFloat4, best for Llama models
    )
    # Load tokenizer and model
    model = AutoAdapterModel.from_pretrained(
        model_id,
        device_map={"": device},
        quantization_config=quant_config,
        token=token,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=token,
        cache_dir=cache_dir
    )

    tokenizer.pad_token = tokenizer.eos_token # For padding in batch

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

def generate_answer(input_text, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(DEVICE)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model.generate(**inputs, temperature=0.1, top_k=10, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return generated_text

def clean_answer(text):
    
    # TODO: Add more extracting logic here
    # Extract only the answer part from generated text
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    # TODO: Add tolerance relative to the size of the number
    # Handle percentage
    percent_match = re.search(r'[-+]?\d*\.?\d+\s*%', text)
    if percent_match:
        number = float(percent_match.group(0).replace('%', '').strip())
        return f"{round(number, 0)}%"

    # TODO: Add tolerance relative to the size of the number
    # Handle regular numbers
    decimal_match = re.search(r'[-+]?\d*\.?\d+', text, re.MULTILINE)
    if decimal_match:
        number = float(decimal_match.group(0))
        # If it's close to an integer, round it
        if abs(round(number) - number) < 0.01:
            return str(round(number))
        # Otherwise, round to one decimal place
        return str(round(number, 0))

    # Handle yes/no answers
    text = text.lower().strip()
    if 'yes' in text or 'true' in text:
        return 'yes'
    if 'no' in text or 'false' in text:
        return 'no'

    return text.strip()


def compute_metrics(p: EvalPrediction):
    predicted_answer = p.predictions[0]
    expected_answer = p.labels[0]

    clean_p = clean_answer(predicted_answer)
    clean_e = clean_answer(expected_answer)

    similarity = fuzz.ratio(clean_p.lower(), clean_e.lower())
    
    return {"acc": int(similarity >= SIMILARITY_THRESHOLD)}

if __name__ == "__main__":
    main()