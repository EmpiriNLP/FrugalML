import json
from dotenv import load_dotenv
import os

load_dotenv()
DATSET_DIR = os.getenv("DATASET_DIR") + "FinQA/dataset/"

sets = ["train", "dev", "test"]
for set in sets:
    data_path = DATSET_DIR + f"{set}.json"
    to_path = DATSET_DIR + f"{set}.cleaned.json"

    with open(data_path, "r") as f:
        data = json.load(f)

    print(f"Original {set} data size: {len(data)}")
    for element in data:
        if element["qa"]["answer"] == "":
            data.remove(element)
        else:
            element["qa"]["exe_ans"] = str(element["qa"]["exe_ans"])

    print(f"Cleaned {set} data size: {len(data)}")

    with open(to_path, "w") as f:
        json.dump(data, f, indent=4)

