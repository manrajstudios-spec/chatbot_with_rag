import numpy as np
import torch
import torch.nn as nn
import torch.optim
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import DataLoader,Dataset
from transformers import get_scheduler
from sklearn.model_selection import StratifiedShuffleSplit
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("../Data/data.json", 'r') as file:
    data = json.load(file)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.35,random_state=42)

X=np.array(data['lines'])
y=np.array(data['labels'])
for t,e in split.split(X,y):
    X_train,X_val = X[t],X[e]
    y_train,y_val = y[t],y[e]

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

model.to(device)

for param in model.distilbert.embeddings.parameters():
    param.requires_grad = False

for i, layer in enumerate(model.distilbert.transformer.layer):
    if i < 4:
        for param in layer.parameters():
            param.requires_grad = False


model.classifier = nn.Sequential(nn.Dropout(0.3),nn.Linear(model.config.dim, 2))
model.classifier.to(device)

class Make_Dataset(Dataset):
    def __init__(self, X, y):
        self.tokens = tokenizer(X.tolist(),truncation=True,padding=True,max_length=64)
        self.labels = y.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]).to(device) for k, v in self.tokens.items()}
        item["labels"] = torch.tensor(self.labels[idx]).to(device)
        return item

train_dataset = DataLoader(Make_Dataset(X_train, y_train),batch_size=8,shuffle=True)
val_dataset = DataLoader(Make_Dataset(X_val, y_val),batch_size=8,shuffle=True)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset)*2)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

for epoch in range(2):
    model.train()

    t_loss =0
    for batch in train_dataset:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        t_loss+= loss.item()

    print(f"epoch = {epoch} loss {t_loss}")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_dataset:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += len(batch['labels'])

        print(f"epoch: {epoch} accuracy: {correct/total}")

model.save_pretrained("../Model/intent_clf")
tokenizer.save_pretrained("../Model/intent_tokenizer")

