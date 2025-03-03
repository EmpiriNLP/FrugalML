import json
import os
from torch.utils.data import Dataset

class FinQADataset(Dataset):
    def __init__(self, file_path):
        # Load the JSON data from file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Directly extract pre_text, table, and post_text from the top-level keys
        pre_text = item.get('pre_text', '')
        table = item.get('table', '')
        post_text = item.get('post_text', '')
        
        # Extract QA information
        qa = item.get('qa', {})
        question = qa.get('question', '')
        program = qa.get('program', '')
        
        # Create the context by concatenating pre_text, table, and post_text
        context = f"{pre_text}\n{table}\n{post_text}".strip()
        
        # Generate the prompt using the given format
        prompt = f"""
        Context:
        {context}

        Given the context, {question} Report your answer using the following format:
        Explanation: Explanation of calculation
        Formatted answer: Number with two decimal point precision and no units
        """.strip()
        
        return {
            'context': context,
            'question': question,
            'label': program,
            'prompt': prompt
        }

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "./dataset/train.json")
    dataset = FinQADataset(file_path)
    sample = dataset[0]
    print("Prompt:\n", sample['prompt'])
    print("Label:\n", sample['label'])
