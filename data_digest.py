import json
import pandas as pd

DATA_FILE = "D:/datasets/EmpiriNLP/FinQA/dataset/train.json"
OUTPUT_FILE = "D:/datasets/EmpiriNLP/FinQA/dataset/train.tsv"

def preprocess_text(strings: list) -> str:
    strings = filter(lambda x: x != ".", strings)
    return " ".join(strings)

def preprocess_table(table: list[list[str]]) -> str:
    return "\n".join(["|".join(row) for row in table])

def json_to_tsv(json_data: dict) -> pd.DataFrame:
    data = []
    for item in json_data:
        data.append({
            "id": item["id"],
            "pre_text": preprocess_text(item["pre_text"]),
            "table": preprocess_table(item["table"]),
            "post_text": preprocess_text(item["post_text"]),
            "question": item["qa"]["question"],
            "answer": item["qa"]["exe_ans"],
        })
    return pd.DataFrame(data)

if "__main__" == __name__:
    json_data = json.load(open(DATA_FILE))
    tsv_data = json_to_tsv(json_data)
    tsv_data.to_csv(OUTPUT_FILE, index=False, sep="\t")