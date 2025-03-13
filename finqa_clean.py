import json

data_path = "D:/datasets/EmpiriNLP/FinQA/dataset/train.json"
to_path = "D:/datasets/EmpiriNLP/FinQA/dataset/train.cleaned.json"

with open(data_path, "r") as f:
    data = json.load(f)

for element in data:
    element["qa"]["exe_ans"] = str(element["qa"]["exe_ans"])

with open(to_path, "w") as f:
    json.dump(data, f, indent=4)

