import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig, DataCollatorWithPadding
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from fuzzywuzzy import fuzz
import re
import time
from tracker import profile_training_flops_by_steps


class CastOutputToBFloat16(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.bfloat16)

def typecast_model_layers(model):
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.bfloat16)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToBFloat16(model.lm_head)

def generate_lora_config(model):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.2,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    return model

def print_trainable_parameters(base_model):
    trainable_params = 0
    all_params = 0
    for _, param in base_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params} || All parameters: {all_params} || Trainable parameters %: {trainable_params/all_params*100}")

def set_training_arguments(model, train_dataset_finqa, val_dataset_finqa, tokenizer):
    training_args = TrainingArguments(
        output_dir="./results/llama_finqa_results",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        warmup_steps=100,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01,
        learning_rate=2e-4,
        logging_dir="./logs",
        logging_steps=10,
        bf16=True,
        report_to="none",
        load_best_model_at_end=True,
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset_finqa,
        eval_dataset=val_dataset_finqa,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    return trainer

@profile_training_flops_by_steps(csv_path="flops_profiler/training_flops_profiler_llama318.csv", log_every_n_steps=50)
def train_model(trainer):
    return trainer.train()

def calculate_accuracy_similarity(reference_answers, generated_answers):
    total_samples = 0
    correct_predictions = 0
    threshold = 85  # Minimum similarity percentage for correct match
    similarity_scores = []
    SETTING = "FinQA"

    for i, generated_answer in enumerate(generated_answers):
        total_samples += 1
        reference_answer = reference_answers[i]
        clean_e = clean_answer(reference_answer)
        clean_p = clean_answer(generated_answer)


        # Compute similarity score
        similarity = fuzz.ratio(clean_p.lower(), clean_e.lower())
        similarity_scores.append(similarity)
        print()
        print()
        print("************************************************************************")
        print (f"*** Input {total_samples} *** ")
    #    print(example["input_text"])
        print ("*** Expected answer *** ")
        print(reference_answer)
        print ("*** Clean Expected answer *** ")
        print(clean_e)
        print ("*** Predicted answer *** ")
        print(generated_answer)
        print ("*** Clean Predicted answer *** ")
        print(clean_p)

        # Count as correct if similarity is above threshold
        if similarity >= threshold:
            correct_predictions += 1

    # Print Results
    accuracy = correct_predictions / total_samples * 100
    avg_similarity = sum(similarity_scores) / total_samples

    print(f"*************************************{SETTING}***********************************")

    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Similarity Score: {avg_similarity:.2f}%")

def clean_answer(text):
    """Extract and format numerical answer"""
    percent_match = re.search(r'[-+]?\d*\.?\d+\s*%', text)
    if percent_match:
        number = float(percent_match.group(0).replace('%', '').strip())
        return f"{round(number, 0)}%"

    # Handle regular numbers
    pre_decimal_match = text
    decimal_match = re.search(r'[-+]?\d*,?\.?\d+', pre_decimal_match, re.MULTILINE)
    if decimal_match:
        number = float(decimal_match.group(0).replace(',', ''))
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

def model_inference(model, tokenizer, test_dataloader_finqa, test_sample_size, device):
    start = time.time()

    model.eval()

    generated_answers = []
    reference_answers = []
    reference_questions = []
    sample_count = 0
    target_count = test_sample_size

    with torch.no_grad():
        for batch in test_dataloader_finqa:
            if sample_count >= target_count:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.shape[0]
            
            # Determine how many samples to take from this batch
            samples_needed = min(batch_size, target_count - sample_count)
            
            # Take only the needed samples from this batch
            batch_input_ids = input_ids[:samples_needed]
            batch_attention_mask = attention_mask[:samples_needed]
            batch_labels = labels[:samples_needed]
            
            # Decode questions
            batch_questions = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

            # Generate answers
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode generated answers
            batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for batch_answer in batch_answers:
                if ("Answer:" in batch_answer):
                    batch_answer = re.search(r'(?<=Answer:\s)\s?.+?(?=\\n|\n|$)', batch_answer).group(0)
                generated_answers.append(batch_answer.strip())
            
            # Add to our collections
            reference_questions.append(batch_questions)
            reference_answer = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
            reference_answer = re.search(r'(?<=\[\').+(?=(\"|\')\])', str(reference_answer))
            if reference_answer:
                reference_answer = reference_answer.group(0)
            else:
                reference_answer = ""
            reference_answers.append(reference_answer)
            
            # Update count
            sample_count += samples_needed

    print(f"Collected {len(reference_questions)} samples:")
    for i, (question, answer) in enumerate(zip(reference_questions, generated_answers)):
        print(f"\nSample {i+1}:")
        print(f"Q: {question}")
        print(f"A: {answer}")

    end = time.time()
    print(f"\nTime taken: {end - start} seconds")

    return generated_answers, reference_answers


def create_prompt_instance(table, pre_text, post_text, question, answer):
    set_context = """
    You are an intelligent financial data analyst. You are given a table with financial data. You are also given a pre-text and a post-text that provide some context about the data in the table. 
    You are asked a question about the data in the table or paragraph. 
    You are expected to answer the question based on the data in the table and the paragraph.
    """
    table_prompt = """
    The table provide the financial data. All the elements in the table are separated by \"|\". 
    The first row of the table contains the column names. In the following rows, the first column contains the row name and the rest of the elements are the values in the row assigned to the respective columns. 
    Interpret the table and use the data in it to calculate the answer to the provided quesions.
    """
    pre_text_prompt = """
    The pre_text provides some context about the data before the table. It may contain information that is not present in the table. 
    It may also contain some numbers which might require arithmatic processing to get the answer. There may be multiple sentences separated by comma (,). 
    Interpret each pre_text paragraph and use the data and description in it to infer the answer to the provided quesions.
    """
    post_text_prompt = """
    The post_text provides some context about the data after the table. It may contain information that is not present in the table. 
    It may also contain some numbers which might require arithmatic processing to get the answer. There may be multiple sentences separated by comma (,). 
    Interpret each post_text paragraph and use the data and description in it to infer the answer to the provided quesions.
    """
    question_prompt = """
    The question is asked based on the data in the table, in the pre-text, and in the post-text. 
    You are expected to answer the question based on this data in the table, in the pre-text, and in the post-text.
    """
    answer_prompt = ""
    answer_instruction_prompt = """
    INSTRUCTIONS:
    - Provide a direct and concise answer to the question based on the data in the table and the pre-text and the post-test
    - Do NOT repeat the pre-text, table, post-text or question in your response
    - Give the answer to the question first based on the given context before providing any explanation
    - Let's think step by step!

    Answer: """

    table_prompt += """
    Table Data:
    """
    for row in table:
        table_prompt += "|".join([str(cell) for cell in row]) + " \n"

    pre_text_prompt += f"""
    Pre Text Data: 
    {pre_text}"""
    
    post_text_prompt += f"""
    Post Text Data: 
    {post_text}
    """
    question_prompt += f"""
    Question: 
    {question}
    """
    answer_prompt = str(answer)

    return set_context, table_prompt, pre_text_prompt, post_text_prompt, question_prompt, answer_prompt, answer_instruction_prompt

class FinQADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.qa_pairs = []

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tables = item.get('table', [])
        pre_text = item.get('pre_text', [])
        post_text = item.get('post_text', [])
        qa = item.get('qa', {})

        question = qa["question"].strip()
        answer = qa["answer"]

        set_context, table_prompt, pre_text_prompt, post_text_prompt, question_prompt, answer_prompt, answer_instruction_prompt = create_prompt_instance(tables, pre_text, post_text, question, answer)
        input_context = (set_context + pre_text_prompt + table_prompt + post_text_prompt + question_prompt + answer_instruction_prompt).strip()
        label_text = answer_prompt.strip()

        inputs = self.tokenizer( 
                    input_context, 
                    truncation=True, 
                )

        labels = self.tokenizer(
                    label_text, 
                    max_length=16,
                    truncation=True, 
                    padding="max_length", 
                )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = labels["input_ids"]


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def clean_dataset(data_finqa):
    data_finqa = [entry for entry in data_finqa if entry.get('qa').get('question')]
    data_finqa = [entry for entry in data_finqa if entry.get('qa').get('answer')]
    return data_finqa

def load_finqa_dataset(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    cache_dir = "/cs/student/projects1/aibh/2024/tpatil/.cache/huggingface/"
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    token = "#######KEY#######"

    quant_config = BitsAndBytesConfig(load_in_4bit=True) 

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir,
        token=token,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2"
    )

    base_model_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
        token=token
    )
    if base_model_tokenizer.pad_token is None:
        base_model_tokenizer.pad_token = base_model_tokenizer.eos_token
        base_model_tokenizer.pad_token_id = base_model_tokenizer.eos_token_id
    base_model_tokenizer.padding_side = 'left'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_finqa = load_finqa_dataset("/cs/student/projects1/aibh/2024/tpatil/comp0087/FrugalML/datasets/FinQA/train.json")
    val_data_finqa = load_finqa_dataset("/cs/student/projects1/aibh/2024/tpatil/comp0087/FrugalML/datasets/FinQA/dev.json")
    test_data_finqa = load_finqa_dataset("/cs/student/projects1/aibh/2024/tpatil/comp0087/FrugalML/datasets/FinQA/test.json")

    train_data_finqa = clean_dataset(train_data_finqa)
    val_data_finqa = clean_dataset(val_data_finqa)
    test_data_finqa = clean_dataset(test_data_finqa)

    train_dataset_finqa = FinQADataset(train_data_finqa, base_model_tokenizer)
    val_dataset_finqa = FinQADataset(val_data_finqa, base_model_tokenizer)
    test_dataset_finqa = FinQADataset(test_data_finqa, base_model_tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=base_model_tokenizer, padding='longest')

    test_dataloader_finqa = DataLoader(
        test_dataset_finqa, 
        batch_size=1, 
        shuffle=False,
        collate_fn=data_collator
    )

    generated_answers, reference_answers = model_inference(
        base_model,
        base_model_tokenizer,
        test_dataloader_finqa,
        len(test_dataset_finqa),
        device=device
    )

    calculate_accuracy_similarity(reference_answers, generated_answers)

    typecast_model_layers(base_model)
    base_model = generate_lora_config(base_model)
    print_trainable_parameters(base_model)

    trainer = set_training_arguments(base_model, train_dataset_finqa, val_dataset_finqa, base_model_tokenizer)
    base_model.train()

    training_output  = train_model(trainer)
    trainer.save_model("./results_finqa/finqa_llama_8_v1")
    trainer.evaluate()


    ft_model_path = "./results_finqa/finqa_llama_8_v5_bs2"
    lora_config = PeftConfig.from_pretrained(ft_model_path)
    ft_model = PeftModel.from_pretrained(
        base_model, 
        ft_model_path, 
        config=lora_config,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2"
    )
    base_model_tokenizer = AutoTokenizer.from_pretrained(
        ft_model_path,
        trust_remote_code=True,
    )

    ft_generated_answers, ft_reference_answers = model_inference(
        ft_model,
        base_model_tokenizer,
        test_dataloader_finqa,
        len(test_dataset_finqa),
        device=device
    )

    calculate_accuracy_similarity(ft_reference_answers, ft_generated_answers)


if __name__ == "__main__":
    main()