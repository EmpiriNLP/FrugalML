import os
import torch
import csv
from functools import wraps
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from torch.utils.data import Dataset
import gc
import torch

gc.collect()
torch.cuda.empty_cache()


import json

class FinQADataset:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = FinQADataset("/cs/student/projects2/aisd/2024/giliev/FinQA/dataset/train.json")

# -----------------------------
# Load tokenizer and fine-tuned model
# -----------------------------
model_path = "/cs/student/projects2/aisd/2024/giliev/FrugalML/prefix_tuned_llama_3_1_8B_100tokens_3epochs_1batch_full"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

quant_config = BitsAndBytesConfig(load_in_4bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",  # don't move it with .to("cuda")
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=quant_config,
    attn_implementation="flash_attention_2",  # optional if supported
)

model = PeftModel.from_pretrained(
    base_model,
    "/cs/student/projects2/aisd/2024/giliev/FrugalML/prefix_tuned_llama_3_1_8B_100tokens_3epochs_1batch_full"
)
model.eval()

# -----------------------------
# Tokenized FinQA Dataset Wrapper
# -----------------------------
class TokenizedFinQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["qa"]["question"].strip()
        program = item["qa"]["program"].strip() if isinstance(item["qa"]["program"], str) else " ".join(item["qa"]["program"])
        answer = item["qa"]["answer"].strip()

        # You can define your prompt/label logic here
        full_text = f"Question: {question}\nProgram: {program}\nAnswer: {answer}"
        tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
tokenized_dataset = TokenizedFinQADataset(train_data, tokenizer)

# -----------------------------
# Dynamic Padding Collator
# -----------------------------
class CustomDataCollatorWithPadding:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        max_label_len = max(len(l) for l in labels)
        padded_labels = [l + [self.label_pad_token_id] * (max_label_len - len(l)) for l in labels]
        batch["labels"] = torch.tensor(padded_labels)
        return batch

collator = CustomDataCollatorWithPadding(tokenizer)

# -----------------------------
# Tracker Decorator
# -----------------------------
def profile_training_flops_by_steps(csv_path="tracker_logs.csv", log_every_n_steps=10):
    def decorator(training_func):
        @wraps(training_func)
        def wrapper(*args, **kwargs):
            trainer = args[0]
            original_training_step = trainer.training_step
            step_counter = [0]

            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(['Step', 'FLOPs', 'CUDA Time (ms)', 'CPU Time (ms)', 'Memory (MB)', 'Loss'])

            def profiled_step(*step_args, **step_kwargs):
                step_counter[0] += 1
                step = step_counter[0]

                if step % log_every_n_steps == 0:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA
                        ],
                        profile_memory=True,
                        with_flops=True
                    ) as prof:
                        result = original_training_step(*step_args, **step_kwargs)

                    total_flops = sum(e.flops for e in prof.key_averages() if hasattr(e, "flops"))
                    cuda_time = sum(e.cuda_time for e in prof.key_averages()) / 1e3  # ms
                    cpu_time = sum(e.cpu_time for e in prof.key_averages()) / 1e3    # ms
                    mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
                    loss = result["loss"].item() if isinstance(result, dict) else float(result)

                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([step, total_flops, cuda_time, cpu_time, mem_mb, loss])

                    torch.cuda.reset_peak_memory_stats()
                    print(f"[Step {step}] FLOPs={total_flops:.2e}, CUDA={cuda_time:.2f}ms, CPU={cpu_time:.2f}ms, Mem={mem_mb:.2f}MB, Loss={loss:.4f}")
                    return result
                else:
                    return original_training_step(*step_args, **step_kwargs)

            trainer.training_step = profiled_step
            try:
                return training_func(*args, **kwargs)
            finally:
                trainer.training_step = original_training_step
        return wrapper
    return decorator

# -----------------------------
# Training Arguments
# -----------------------------
args = TrainingArguments(
    output_dir="./tracker_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=400,
    learning_rate=5e-5,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    fp16=True
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

# -----------------------------
# Execute Training with Tracker
# -----------------------------
@profile_training_flops_by_steps(csv_path="training_tracker_metrics.csv", log_every_n_steps=10)
def run_training(trainer):
    return trainer.train()

run_training(trainer)
