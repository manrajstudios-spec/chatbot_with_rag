import json
import random
"""

lines = positive + negative
labels = [1] * len(positive) + [0] * len(negative)

combined = list(zip(lines, labels))
random.shuffle(combined)

lines, labels = zip(*combined)

dataset = {
    "lines": list(lines),
    "labels": list(labels)
}

with open("/home/manraj_studios/PycharmProjects/Yuzu-Ai-Companion/Data/data.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)


print("positive:", len(positive))
print("negative:", len(negative))
print("total:", len(dataset["lines"]))

assert len(dataset["lines"]) == len(dataset["labels"])

print("dataset valid")
print("saved -> intent_dataset.json")
"""