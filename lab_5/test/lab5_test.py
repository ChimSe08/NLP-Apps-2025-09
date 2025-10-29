# -*- coding: utf-8 -*-
"""
Baseline: LogisticRegression + TFIDF (1-gram)
- Tự load data từ data/imdb_50k.csv nếu có; else dùng demo_data (~100 câu).
- In: độ chính xác, F1, ROC-AUC (nếu nhị phân), classification report,
      ma trận nhầm lẫn, 10 mẫu dự đoán, top n-grams (feature importance).
"""

import os, csv, random, re
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

random.seed(42)
np.random.seed(42)

# ---------- Data ----------
def _exists_large_csv():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "imdb_50k.csv")
    path = os.path.abspath(path)
    return path if os.path.isfile(path) else None

def _read_csv_guess_cols(csv_path: str, limit: int = 6000):
    """Đọc CSV có thể có cột (text,label) hoặc (review,sentiment)."""
    texts, labels = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.lower() for c in reader.fieldnames]
        tcol = "text" if "text" in cols else "review"
        lcol = "label" if "label" in cols else "sentiment"
        for i, row in enumerate(reader):
            if limit and i >= limit: break
            txt = row.get(tcol) or row.get(tcol.title()) or row.get("text")
            lab = row.get(lcol) or row.get(lcol.title()) or row.get("label")
            if txt is None or lab is None: continue
            try:
                y = int(lab)
            except:
                y = 1 if str(lab).strip().lower() in {"pos","positive","1","true"} else 0
            texts.append(txt)
            labels.append(y)
    return texts, np.array(labels, dtype=int)

def _demo_data() -> Tuple[List[str], np.ndarray]:
    pos = [
        "I absolutely loved this movie, it was fantastic!",
        "What a wonderful experience, highly recommend.",
        "Great performances and beautiful soundtrack.",
        "The plot was tight and the ending was satisfying.",
        "A charming cast and witty dialog kept me smiling.",
        "Surprisingly emotional and very well directed.",
        "The visuals are stunning and the pacing is great.",
        "Heartwarming story with believable characters.",
        "Solid film with strong acting and clever twists.",
        "Brilliant! I would watch it again.",
    ]
    neg = [
        "Terrible plot and bad acting. Waste of time.",
        "The film was boring and way too long.",
        "I wouldn't watch it again; it's awful.",
        "Predictable story and unfunny jokes.",
        "Messy editing and confusing scenes.",
        "Weak script and painfully slow pacing.",
        "Cheap effects and wooden performances.",
        "Cringe dialogue; I almost fell asleep.",
        "Disappointing and poorly executed.",
        "One of the worst movies I've seen.",
    ]
    # mở rộng ~100 câu
    aug = []
    for s in pos: aug.append((s,1))
    for s in neg: aug.append((s,0))
    # thêm nhiễu nhẹ
    extra = []
    for (s,y) in aug:
        s2 = re.sub(r'[!.]', '', s.lower())
        extra.append((s2, y))
    dataset = aug + extra + [(s+"!",y) for (s,y) in aug]
    random.shuffle(dataset)
    X = [t for t,_ in dataset]
    y = np.array([lab for _,lab in dataset], dtype=int)
    return X, y

def load_data():
    csv_path = _exists_large_csv()
    if csv_path:
        X, y = _read_csv_guess_cols(csv_path, limit=6000)  # đủ lớn nhưng vẫn chạy nhanh
        print(f"== Using CSV: {csv_path}  | n={len(X)}")
    else:
        X, y = _demo_data()
        print(f"== Using built-in demo dataset | n={len(X)}")
    return X, y

# ---------- Train / Eval ----------
def evaluate(name, y_true, y_pred, proba1=None):
    print(f"\n=== {name} ===")
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="binary")
    out = f"Accuracy: {acc:.4f} | F1: {f1:.4f}"
    if proba1 is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, proba1)
            out += f" | ROC-AUC: {auc:.4f}"
        except Exception:
            pass
    print(out)
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

def top_features_tfidf(pipe: Pipeline, k=15):
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"): return
    feats = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top_pos = feats[np.argsort(coefs)[-k:][::-1]]
    top_neg = feats[np.argsort(coefs)[:k]]
    print("\nTop + coefficients:", ", ".join(top_pos))
    print("Top - coefficients:", ", ".join(top_neg))

def main():
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), max_features=20000)),
        ("clf",   LogisticRegression(max_iter=1000, n_jobs=None))
    ])
    pipe.fit(Xtr, ytr)
    proba1 = None
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba1 = pipe.predict_proba(Xte)[:,1]
    yhat = pipe.predict(Xte)
    evaluate("Baseline (LogReg + TFIDF 1-gram)", yte, yhat, proba1)
    top_features_tfidf(pipe, 15)

    # In 10 mẫu dự đoán
    print("\nSample predictions:")
    for i in range(min(10, len(Xte))):
        p = proba1[i] if proba1 is not None else None
        pstr = f"{p:.4f}" if p is not None else "NA"
        print(f"  y_true={yte[i]}  y_pred={yhat[i]}  prob1={pstr}  |  {Xte[i][:80]}")

    # K-Fold nhanh
    print("\n=== 5-Fold quick CV ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        pipe.fit([X[i] for i in tr], y[tr])
        yk = pipe.predict([X[i] for i in te])
        f1s.append(f1_score(y[te], yk))
        print(f"Fold {k}: F1={f1s[-1]:.4f}")
    print(f"F1 mean={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    main()
