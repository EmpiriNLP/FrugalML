import json
from dotenv import load_dotenv
import os

load_dotenv()
DATSET_DIR = os.getenv("DATASET_DIR") + "FinQA/dataset/"

data_path = DATSET_DIR + "train.json"
to_path = DATSET_DIR + "train.cleaned.json"

with open(data_path, "r") as f:
    data = json.load(f)

for element in data:
    element["qa"]["exe_ans"] = str(element["qa"]["exe_ans"])

with open(to_path, "w") as f:
    json.dump(data, f, indent=4)

