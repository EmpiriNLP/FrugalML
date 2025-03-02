import torch
from datasets import Features, Value, load_dataset, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fuzzywuzzy import fuzz
import re

# Load Llama-3.2-1B
# MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

SETTING = ""

# Define 4-bit quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation
    bnb_4bit_use_double_quant=True,  # Double quantization for better memory efficiency
    bnb_4bit_quant_type="nf4"  # NormalFloat4, best for Llama models
)

# Load tokenizer and model
#model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": DEVICE})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": DEVICE},
    quantization_config=quant_config
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess_function(example):
    # Ensure 'qa' exists and has 'question' and 'answer'
    if "qa" not in example or not isinstance(example["qa"], dict):
        return None  # Skip invalid entries

    input_text = f"{example['pre_text']} {example['post_text']} {example['table']}"
    question = example['qa'].get("question", "").strip()
    expected_answer = str(example['qa'].get("answer", "")).strip()  # Force text

    return {
        "input_text": f"Context: {input_text}\nQuestion: {question}\nAnswer:",
        "expected_answer": expected_answer
    }

def preprocess_function2(example):
    question = example["qa"].get("question", "No question available.")
    expected_answer = str(example['qa'].get("answer", "")).strip()  # Force text
    table = example.get("table", [])
    table_str = "\n".join([" | ".join(row) for row in table])

    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))

#    gold_inds = example["qa"].get("gold_inds", {})
#    relevant_info = "\n".join(gold_inds.values())
#    program = example["qa"].get("program", "")

    input_text = (
        "You are a financial calculator. Follow these steps:\n"
        "Return ONLY the final numerical answer with no text explanation\n\n"
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

# Load dataset
features = Features({
    "pre_text": Value("string"),
    "post_text": Value("string"),
    "qa": Features({
        "question": Value("string"),
        "answer": Value("string")
    })
})
print ("*** Data set loading *** ")
dataset = load_dataset(
    "json",
    data_files="train3.json",
    split="train",
#    features=features
)
print ("*** Data set loaded *** ")

# Apply preprocessing
dataset = dataset.map(preprocess_function2, remove_columns=dataset.column_names)

# Function to run inference
def generate_answer_old(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, temperature=0.1, top_k=10)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer part from generated text
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:")[-1].strip()

    return generated_text

def generate_answer(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    input_ids = inputs["input_ids"]  # Explicitly access input_ids tensor

    with torch.no_grad():
        outputs = model.generate(**inputs, temperature=0.1, top_k=10, max_new_tokens=20)
    #generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Extract only the answer part from generated text
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:")[-1].strip()

    return generated_text

# Generate output without repeating the prompt
# output_ids = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
# response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)


def clean_answer(text):
    """Extract and format numerical answer"""
#    if ':' in text:
#        text = text.split(':')[-1]

    # Handle percentages first
    # percent_match = re.search(r'[-+]?\d*\.?\d+\s*%?', text)
    percent_match = re.search(r'[-+]?\d*\.?\d+\s*%', text)
    if percent_match:
        number = float(percent_match.group(0).replace('%', '').strip())
        # If the number is small (likely decimal), convert to percentage
#        if abs(number) < 1:
#            number *= 100
        # Round to one decimal place and add % symbol
        return f"{round(number, 0)}%"

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

# Compute accuracy using fuzzy matching
total_samples = 0
correct_predictions = 0
threshold = 85  # Minimum similarity percentage for correct match
similarity_scores = []

import time

start = time.time()

for example in dataset:
    total_samples += 1
    predicted_answer = generate_answer(example["input_text"])
    expected_answer = example["expected_answer"]
    clean_p = clean_answer(predicted_answer)
    clean_e = clean_answer(expected_answer)


    # Compute similarity score
    # similarity = fuzz.ratio(predicted_answer.lower(), expected_answer.lower())
    similarity = fuzz.ratio(clean_p.lower(), clean_e.lower())
    similarity_scores.append(similarity)
    print()
    print()
    print("************************************************************************")
    print (f"*** Input {total_samples} *** ")
#    print(example["input_text"])
    print ("*** Expected answer *** ")
    print(expected_answer)
    print ("*** Clean Expected answer *** ")
    print(clean_e)
    print ("*** Predicted answer *** ")
    print(predicted_answer)
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

# Block of code to time
end = time.time()

print("Elapsed time:", end - start, "seconds")
print("Avg Inf time:", (end - start)/dataset.shape[0], "seconds")
