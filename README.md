## KẾT QUẢ THỰC HIỆN
Độ chính xác trên tập dev:
0.8683
• Ví dụ dự đoán câu mới:
– Câu: “I love NLP”
– Dự đoán:
```
I       → PRON
love    → VERB
NLP     → PROPN
```

# BÁO CÁO KẾT QUẢ
Độ chính xác trên tập train và dev sau mỗi epoch
Bảng kết quả huấn luyện:
```
| Epoch | Train Acc | Dev Acc | Loss   |
| ----- | --------- | ------- | ------ |
| 1     | 0.7710    | 0.7544  | 1.1288 |
| 2     | 0.8352    | 0.8083  | 0.6273 |
| 3     | 0.8759    | 0.8372  | 0.4714 |
| 4     | 0.9009    | 0.8544  | 0.3722 |
| 5     | 0.9205    | 0.8683  | 0.3046 |
```
Độ chính xác cuối cùng trên tập dev
```
Best Dev Accuracy: 0.8683
```
Hàm predict_sentence(sentence)
Hàm dùng để nhận câu mới dạng chuỗi, chuyển token → index → model để dự đoán POS và in ra từng cặp (từ, nhãn_dự_đoán):
```
def predict_sentence(sentence, model, word_to_ix, ix_to_tag):
    model.eval()
    tokens = sentence.strip().split()
    unk = word_to_ix["<UNK>"]

    ids = [word_to_ix.get(w, unk) for w in tokens]
    x = torch.tensor(ids).unsqueeze(0).to(device)
    lengths = torch.tensor([len(tokens)]).to(device)

    with torch.no_grad():
        logits = model(x, lengths)
        preds = logits.argmax(dim=-1).squeeze(0)

    for w, t in zip(tokens, preds):
        print(f"{w}\t{ix_to_tag[int(t)]}")
```

