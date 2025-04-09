import json
from dotenv import load_dotenv
import os

load_dotenv()
DATASET_DIR = os.getenv("DATASET_DIR") + "FinQA/dataset/"

sets = ["train", "dev", "test"]
for set in sets:
    data_path = DATASET_DIR + f"{set}.json"
    to_path = DATASET_DIR + f"{set}.cleaned.json"

    with open(data_path, "r") as f:
        data = json.load(f)

    print(f"Original {set} data size: {len(data)}")
    cleaned_data = []
    for element in data:
        # Only keep ["qa"]["question"], ["qa"]["answer"], ["table"], ["pre_text"], ["post_text"] if exists
        # Remove if ["qa"]["question"] or ["qa"]["answer"] is empty
        if not element["qa"]["question"].strip() or not element["qa"]["answer"].strip():
            continue

        cleaned_element = {
            "qa": {
                "question": str(element["qa"]["question"]).strip(),
                "answer": str(element["qa"]["answer"]).strip(),
            },
            "table": element.get("table", []),
            "pre_text": element.get("pre_text", []),
            "post_text": element.get("post_text", [])
        }
        cleaned_data.append(cleaned_element)

    print(f"Cleaned {set} data size: {len(cleaned_data)}")

    with open(to_path, "w") as f:
        json.dump(cleaned_data, f, indent=4)
