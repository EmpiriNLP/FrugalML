import torch
from datasets import Features, Value, load_dataset, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, EvalPrediction
from fuzzywuzzy import fuzz
import re
import os
from dotenv import load_dotenv
import logging
import time
from adapters import AutoAdapterModel, AdapterTrainer, LlamaAdapterModel
import sys
import pandas as pd
import itertools
import gc
import json

logging.basicConfig(level=logging.INFO)
load_dotenv()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

ACCESS_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
MODEL_DIR = os.getcwd() + "/models"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
DATASET_DIR = os.getenv("DATASET_DIR") + "FinQA/dataset/"
RESULTS_DIR = os.getenv("RESULTS_DIR") + "FinQA/"
SIMILARITY_THRESHOLD = 85  
RESULTS_HEADER = ["model", "experiment", "trained_samples", "batch_size", "learning_rate", "epochs", "evaluated_samples", "accuracy", "avg_similarity", "total_time", "avg_time_per_sample", "training_time"]
# Other possible headers: "similarity_threshold", "max_new_tokens", etc.

EXPERIMENT = "adapter" # "adapter" or "baseline"


def main(epochs: int =1, batch_size: int =1, number_of_training_samples: int =1, learning_rate: float = 1e-4):
    torch.cuda.empty_cache()
    model, tokenizer = load_model(MODEL_ID, DEVICE, ACCESS_TOKEN, MODEL_DIR)
    train_dataset = load_preprocessed_dataset("json", data_files=DATASET_DIR+"/train.cleaned.json", split="train")
    val_dataset = load_preprocessed_dataset("json", data_files=DATASET_DIR+"/dev.cleaned.json", split="train")

    # epochs = [1, 5, 15]
    # batch_size = [1, 2, 4]
    # number_of_training_samples = [1, 200, 1000]

    # epochs = 15
    # batch_size = 1
    # number_of_training_samples = 200

    if EXPERIMENT == "adapter":
        
        def preprocess_to_finetune(batch):
            prompt = batch["input_text"]
            answer = batch["expected_answer"]
            full_text = [p+a for p, a in zip(prompt, answer)]

            full_text_encodings = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=4096)

            prompt_encodings = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            prompt_len = prompt_encodings["input_ids"].shape[1] # One length is enough if padding + truncation

            labels = full_text_encodings["input_ids"].clone()
            labels[:, :prompt_len] = -100 # Ignore loss on prompt tokens?

            full_text_encodings["labels"] = labels # Tokenized_full has input_ids, attention_mask, labels

            return  full_text_encodings
            # encoded = tokenizer(batch["input_text"], truncation=True, padding=True)
            # return  encoded


        train_dataset = train_dataset.map(preprocess_to_finetune, batched=True, batch_size=batch_size)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], device=DEVICE) 
        val_dataset = val_dataset.map(preprocess_to_finetune, batched=True, batch_size=batch_size)
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], device=DEVICE)

        if number_of_training_samples != -1:
            train_dataset = train_dataset.select(range(number_of_training_samples))

        val_dataset = val_dataset.select(range(100)) # For significant loss

        model.add_adapter("finance_adapter", config="seq_bn")
        model.add_causal_lm_head("finance_adapter")
        model.train_adapter("finance_adapter")

        # Adapter layers are added outside the model's original layers, so manually convert them to bfloat16 and move to device
        for name, param in model.named_parameters():
            if "finance_adapter" in name:
                param.data = param.data.to(torch.bfloat16)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(torch.bfloat16)
                    
        model.to(DEVICE)

        # Clean output dir before training
        if os.path.exists("./models/adapter"):
            os.system("rm -rf ./models/adapter")
            logging.info("Cleaned output directory")

        training_args = TrainingArguments(
            learning_rate=learning_rate,
            warmup_steps=100,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            output_dir="./models/adapter",
            overwrite_output_dir=True,
            eval_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            # save_steps=10,
            # disable_tqdm=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # label_smoothing_factor=0.1, # To enable default label smoothing loss function
            save_strategy="epoch",
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            # metric_for_best_model="token_accuracy",
            # greater_is_better=False,
            load_best_model_at_end=True,
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # compute_metrics=compute_accuracy,
            # compute_loss_func=compute_loss
        )

        logging.info("Starting Training")
        start = time.time()
        trainer.train()
        end = time.time()
        training_time = round(end - start, 4)

        # trainer.evaluate()

        # Move a copy of "trainer_state.json" from "./models/adapter/checkpoint-*/" to results directory
        checkpoint_folders = os.listdir("./models/adapter")
        if len(checkpoint_folders) > 1:
            logging.warning("Multiple checkpoint folders found, using the latest one")
        checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x.split("-")[-1]), reverse=True)
        checkpoint_folder = os.path.join("./models/adapter", checkpoint_folders[0])
        trainer_state_path = os.path.join(checkpoint_folder, "trainer_state.json")
        state_path = os.path.join(RESULTS_DIR, f"states.ts_{number_of_training_samples}-bs_{batch_size}-lr_{learning_rate}.json")
        if os.path.exists(trainer_state_path):
            # os.rename(trainer_state_path, state_path)
            # logging.info(f"Trainer state saved to {state_path}")
            os.system(f"cp {trainer_state_path} {state_path}")
            logging.info(f"Trainer state copied to {state_path}")
        else:
            logging.warning(f"Trainer state not found in {checkpoint_folder}")


    # Evaluate the model
    # Print model dtype
    logging.info(f"Model dtype: {model.dtype}")
    # Turn adapter layers to bfloat16
    for name, param in model.named_parameters():
        if "finance_adapter" in name:
            param.data = param.data.to(torch.bfloat16)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(torch.bfloat16)

    results_path = os.path.join(RESULTS_DIR, "results.tsv")
    if not os.path.exists(results_path):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_df = pd.DataFrame(columns=RESULTS_HEADER)
        results_df.to_csv(results_path, index=False, sep="\t")

    test_dataset = load_preprocessed_dataset("json", data_files=DATASET_DIR+"/test.cleaned.json", split="train")

    number_of_test_samples = len(test_dataset)

    answer_output_path = os.path.join(RESULTS_DIR, f"answers.ts_{number_of_training_samples}-bs_{batch_size}-lr_{learning_rate}.json") if EXPERIMENT == "adapter" else os.path.join(RESULTS_DIR, f"answers.temp.json")
    results = evaluate_model(model, tokenizer, test_dataset, answers_output_path=answer_output_path)
    results["model"] = MODEL_ID
    results["experiment"] = EXPERIMENT
    if results["experiment"] == "baseline":
        results["trained_samples"] = 0
        results["training_time"] = 0
        results["epochs"] = 0
        results["batch_size"] = 0
        results["learning_rate"] = 0
    else:
        results["experiment"] = "adapter"
        results["trained_samples"] = number_of_training_samples
        results["training_time"] = training_time
        results["epochs"] = epochs
        results["batch_size"] = batch_size
        results["learning_rate"] = learning_rate
    results["evaluated_samples"] = number_of_test_samples

    results_df = pd.read_csv(results_path, sep="\t")
    results_df.loc[len(results_df)] = results
    results_df.to_csv(results_path, index=False, sep="\t")
    logging.info(f"Results saved to {results_path}")


    # model = None
    # tokenizer = None
    # gc.collect()
    # del model
    # del tokenizer
    # torch.cuda.empty_cache()
    # logging.info("Model and tokenizer deleted")


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
    if EXPERIMENT == "adapter":
        model = LlamaAdapterModel.from_pretrained(
            model_id,
            device_map={"": device},
            quantization_config=quant_config,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    elif EXPERIMENT == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": device},
            quantization_config=quant_config,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unknown experiment type thus can't decide model: {EXPERIMENT}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=token,
        cache_dir=cache_dir
    )

    tokenizer.pad_token = tokenizer.eos_token # For padding in batch

    return model, tokenizer

def preprocess_function(example):
    # TODO: should raise error or warning when absent
    question = example["qa"]["question"]
    expected_answer = example['qa']["answer"]
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
    dataset = dataset.with_format("torch")  # Use lazy loading
    return dataset

def generate_answer(input_text, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(DEVICE)
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)  # Ensure input IDs are of type torch.long
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


def compute_accuracy(p: EvalPrediction):
    # (b, 881, 128256), which is (batch, sequence, vocab)
    predicted_answers = p.predictions
    # (b, 881), where majority are -100, masked input tokens, only the last or last few tokens are token ids
    expected_answers = p.label_ids

    # Mask out the -100 labels from predicted_answer and expected_answer
    mask = expected_answers != -100
    predicted_answers = predicted_answers[mask] 
    expected_answers = expected_answers[mask]

    # Calculate accuracy as 0/1 with exact match, over the entire batch
    correct_predictions = 0
    print("*"*20)
    for i in range(predicted_answers.shape[0]):
        predicted_answer_ids = torch.argmax(torch.from_numpy(predicted_answers[i]), dim=-1)
        expected_answer_ids = torch.Tensor([expected_answers[i]])

        # Compare the predicted answer with the expected answer
        if torch.equal(predicted_answer_ids, expected_answer_ids):
            correct_predictions += 1

        print(f"Predicted: {predicted_answer_ids}, \nExpected: {expected_answer_ids}")
    print("*"*20)

    # Calculate accuracy
    accuracy = correct_predictions / len(predicted_answers)

    return {
        "token_accuracy": accuracy,
        "avg_mask_length": mask.sum().item() / len(mask),
        "avg_answer_length": (expected_answers != -100).sum().item() / len(expected_answers),
    }

def compute_loss(outputs, labels, num_items_in_batch):
    # Calculate loss as 0/1 loss with exact match
    # outputs["last_hidden_state"].shape = torch.Size([1, 881, 4096])
    # outputs["past_key_values"] shape is (32, 2, 1, 8, 881, 128), probably not needed for loss
    # labels.shape = torch.Size([1, 881]), where majority are -100, masked input tokens, only the last or last few tokens are token ids
    # And the loss should be able to propagate backwards

    return 0
    
def evaluate_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, eval_dataset: torch.utils.data.Dataset, answers_output_path: str=None):
    number_of_samples = len(eval_dataset)
    correct_predictions = 0
    threshold = SIMILARITY_THRESHOLD  
    similarity_scores = []

    if answers_output_path:
        answer_output = []

    logging.info("Starting Inference")
    start = time.time()
    for i, example in enumerate(eval_dataset):

        predicted_answer = generate_answer(example["input_text"], tokenizer, model)
        expected_answer = example["expected_answer"]
        
        if answers_output_path:
            answer_output.append({
                "Question": example["input_text"],
                "Predicted Answer": predicted_answer,
                "Expected Answer": expected_answer
            })

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

    if answers_output_path:
        with open(answers_output_path, "w") as f:
            json.dump(answer_output, f, indent=4)
        logging.info(f"Answers saved to {answers_output_path}")

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

    return {
        "number_of_samples": number_of_samples,
        "accuracy": round(accuracy, 2),
        "avg_similarity": round(avg_similarity, 2),
        "total_time": round(elapsed_time, 4),
        "avg_time_per_sample": round(avg_elapsed_time, 4),
    }

if __name__ == "__main__":

    # Cartesian combinations
    # epochs = [15]
    # batch_size = [4]
    # number_of_training_samples = [1000]
    # learning_rates = [3e-5, 1e-3]
    # combinations = list(itertools.product(number_of_training_samples, batch_size, learning_rates, epochs))

    # Manual combinations
    combinations = [
        (-1, 1, 3e-5, 15),
        (-1, 2, 1e-4, 15),
        (-1, 2, 3e-5, 15),
        (-1, 4, 1e-4, 15),
        (-1, 4, 3e-5, 15),
    ]

    # excluded_combinations = [
    #     (200, 1, 1e-4, 15),
    #     (200, 1, 3e-5, 15),
    #     (200, 1, 1e-3, 15),
    # ]

    for n, b, l, e in combinations:
        # if (n, b, l, e) in excluded_combinations:
        #     continue
        logging.info("===================================")
        logging.info("===================================")
        logging.info(f"Running experiment with epochs={e}, batch_size={b}, number_of_training_samples={n}, learning_rate={l}")
        main(epochs=e, batch_size=b, number_of_training_samples=n, learning_rate=l)
        logging.info("Experiment finished")
        logging.info("===================================")
        logging.info("===================================")
        logging.info("===================================")
