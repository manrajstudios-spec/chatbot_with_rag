import torch
import loader

def get_result(text_to_evaluate):
    loader.model.eval()

    tokens = loader.tokenizer([text_to_evaluate],padding=True,truncation=True,max_length=64,return_tensors="pt")

    with torch.no_grad():
        out = loader.model(input_ids=tokens["input_ids"],attention_mask=tokens["attention_mask"])

    pred = out.logits.argmax(dim=1).item()

    return pred == 1

if __name__=="__main__":
    print(get_result("coding hey "))