import json
import os
from torch.utils.data import Dataset

class FinQADataset(Dataset):
    def __init__(self, json_file, tokenizer=None, max_length=512):
        """
        Initializes the FinQA dataset.

        Args:
            json_file (str): Path to the JSON file (train.json or test.json).
            tokenizer (callable, optional): A function to tokenize the text input.
                For example, a tokenizer from Hugging Face Transformers.
                If None, the raw question string will be returned.
            max_length (int, optional): Maximum length for tokenization. Defaults to 512.
        """
        file_path = os.path.join(os.path.dirname(__file__), json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Extract the question text. Adjust the key if your JSON uses a different field.
        question = item.get("question", "")
        # Extract the label (e.g., answer) if available; test set items may not have a label.
        label = item.get("answer", None)

        if self.tokenizer:
            # Tokenize the question. The tokenizer should return a dictionary containing tensors.
            inputs = self.tokenizer(
                question,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            # Remove the batch dimension (from shape [1, seq_len] to [seq_len]).
            inputs = {key: tensor.squeeze(0) for key, tensor in inputs.items()}
        else:
            inputs = question

        return {"inputs": inputs, "label": label}


if __name__ == "__main__":
    # Example usage:
    # If you plan to use a Hugging Face tokenizer, import it and load one.
    from transformers import AutoTokenizer

    # Initialize a tokenizer (change the model name as needed).
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create dataset instances for training and testing.
    train_dataset = FinQADataset("train.json", tokenizer=tokenizer)
    test_dataset = FinQADataset("test.json", tokenizer=tokenizer)

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of test examples: {len(test_dataset)}")

    # Display a sample from the training dataset.
    sample = train_dataset[0]
    print("\nSample training data:")
    print("Tokenized Input IDs:", sample["inputs"]["input_ids"])
    print("Attention Mask:", sample["inputs"]["attention_mask"])
    print("Label:", sample["label"])
