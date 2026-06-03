import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("/home/manraj_studios/PycharmProjects/Yuzu-Ai-Companion/Model/intent_tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("/home/manraj_studios/PycharmProjects/Yuzu-Ai-Companion/Model/intent_clf")


def get_result(text_to_evaluate):
    model.eval()
    tokens = tokenizer([text_to_evaluate],padding=True,truncation=True,max_lenght=64)
    out = model(
        input_ids=torch.tensor(tokens["input_ids"]),
        attention_mask=torch.tensor(tokens["attention_mask"]))

    pred = out.logits.argmax(dim=1).item()

    return pred == 1

if __name__=="__main__":
    print(get_result("coding hey "))