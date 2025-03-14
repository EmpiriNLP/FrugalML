from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, AdapterConfig
from datasets import load_dataset

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Define adapter configuration
config = AdapterConfig(
    non_linearity="relu",  # Non-linearity for adapter layers
    reduction_factor=2,    # Reduction factor for dimensionality
)

# Create PEFT model, which freezes the base model and trains only adapter parameters
peft_model = get_peft_model(model, config)

# Load and preprocess your FinQA dataset (using your existing code)
train_dataset = load_dataset("json", data_files="path/to/train.cleaned.json", split="train")
train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)

# Set up training (you'll need to define your training loop, e.g., using Trainer)
# Example: Ensure only adapter parameters are trained
peft_model.print_trainable_parameters()  # Check trainable parameters

# Train the model (integrate with your existing training setup)