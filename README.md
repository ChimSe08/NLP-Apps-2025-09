# 1. Giải thích code 
Bài 1 – Masked Language Modeling

Mục tiêu: Dự đoán từ bị che trong câu.
```
mask_filler = pipeline("fill-mask")
predictions = mask_filler("Hanoi is the <mask> of Vietnam.", top_k=5)
```

Giải thích:

pipeline("fill-mask"): tải mô hình phù hợp (DistilRoberta-base).

<mask>: token dùng bởi Roberta.

top_k=5: lấy 5 dự đoán tốt nhất.

Mỗi dự đoán gồm: token_str, score, và câu hoàn chỉnh.

Bài 2 – Text Generation

Mục tiêu: Sinh văn bản tiếp theo.
```
generator = pipeline("text-generation")
output = generator(prompt, max_length=50)
```

Giải thích:

Pipeline tự tải GPT-2.

max_length: độ dài tối đa bao gồm cả prompt.

GPT dựa trên mô hình unidirectional → dự đoán token tiếp theo.

Bài 3 – Sentence Embedding bằng BERT

Mục tiêu: Lấy vector 768 chiều biểu diễn câu.
```
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
last_hidden = outputs.last_hidden_state
```

Giải thích:

last_hidden_state: (batch, seq_len, hidden_size=768)

Mean Pooling:

mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
sentence_embedding = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1)


→ loại bỏ token padding và lấy trung bình các embedding token.
# 2. Phân tích kết quả
Bài 1 – MLM

Kết quả:
```
'capital' — score: 0.9341
```

→ Mô hình dự đoán chính xác với độ tin cậy rất cao.
Điều này hợp lý vì Roberta được huấn luyện với Masked Language Modeling.

Nhận xét:

Dự đoán chính xác và tự nhiên

Mô hình hiểu bối cảnh hai chiều (bidirectional)

Bài 2 – Text Generation

Dạng output:

The best thing about learning NLP is that you can learn it from other people...

Nhận xét:

GPT-2 sinh văn bản trôi chảy, đúng chủ đề

Một số câu lặp lại → đặc điểm thường gặp của GPT-2 (không phải GPT-3/4)

Có coherence ngắn hạn, nhưng không giữ được coherence dài

Bài 3 – Embedding

Kết quả:
```
sentence_embedding.shape = torch.Size([1, 768])
```

Nhận xét:

Vector 768 chiều → đúng hidden size của BERT-base

Mean Pooling hoạt động tốt (không bị ảnh hưởng bởi padding)

Vector này có thể dùng cho các bài toán:

semantic similarity

clustering

classification

# 3. Khó khăn và giải pháp
1. Lỗi không nhận token [MASK]

Nguyên nhân: pipeline mặc định load DistilRoberta → token đúng là <mask>, không phải [MASK].

Giải pháp: thay bằng <mask> hoặc chỉ định model "bert-base-uncased".

2. GPT-2 cảnh báo về max_length và max_new_tokens

Mặc định GPT-2 có generation_config.

Giải pháp: đặt max_new_tokens hoặc max_length rõ ràng.

3. In PDF trên Colab bị mất dòng

Colab cuộn code → khi in PDF bị cắt.

Giải pháp: thêm CSS mở scroll như phần hướng dẫn, hoặc tải HTML để in.

4. Load mô hình chậm

GPT-2 và Roberta khá nặng.

Giải pháp: chạy Colab GPU hoặc tải model trước.
