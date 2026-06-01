from datasets import load_dataset
import json
ds = load_dataset("math-ai/aime24", "default")

data = []
for example in ds["test"]:
    print(example["problem"])
    print(example["answer"])
    answer = example["answer"]
    data.append({
        "prompt": [
            {
                "from": "user",
                "value": example["problem"]
            }
        ],
        "final_answer": answer
        })
with open("aime2024.json", "w") as f:
    json.dump(data, f,indent=4)
