# -*- coding: utf-8 -*-
"""
Improvement #3: TFIDF + LinearSVC (hinge) + GridSearch
- In báo cáo tương tự, thêm CV.
"""
import os, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# ---------- Đường dẫn CSV ----------
def _csv():
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "imdb_50k.csv"))
    return p if os.path.isfile(p) else None

# ---------- Đọc CSV ----------
def _read(p, limit=6000):
    import csv as _csv
    X, y = [], []
    with open(p, encoding="utf-8") as f:
        r = _csv.DictReader(f)
        cols = [c.lower() for c in r.fieldnames]
        tcol = "text" if "text" in cols else "review"
        lcol = "label" if "label" in cols else "sentiment"
        for i, row in enumerate(r):
            if i == limit: break
            t = row.get(tcol) or row.get(tcol.title())
            l = row.get(lcol) or row.get(lcol.title())
            if t is None or l is None:
                continue
            try:
                yv = int(l)
            except:
                yv = 1 if str(l).lower() in {"pos", "positive", "1", "true"} else 0
            X.append(t)
            y.append(yv)
    return X, np.array(y, int)

# ---------- Demo nhỏ: đảm bảo |X| == |y| ----------
def _demo(n=120):
    base_X = ["good fun movie", "really bad boring", "excellent acting great plot", "terrible writing"]
    base_y = [1, 0, 1, 0]
    # nhân/cắt đồng bộ tới n mẫu
    reps = max(1, int(np.ceil(n / len(base_X))))
    X = (base_X * reps)[:n]
    y = np.array((base_y * reps)[:n], int)
    return X, y

def load_data():
    p = _csv()
    if p:
        X, y = _read(p)
        print(f"== Using CSV | n={len(X)}")
    else:
        X, y = _demo(n=120)
        print("== Using demo | n=", len(X))
    # an toàn: ép độ dài khớp nếu có lỗi nhập liệu
    if len(X) != len(y):
        m = min(len(X), len(y))
        X, y = X[:m], y[:m]
        print(f"!! Adjusted lengths to match: n={m}")
    return X, y

def main():
    X, y = load_data()
    # tách train/test
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words='english')),
        ("clf", LinearSVC())
    ])
    grid = {
        "tfidf__max_features": [2000, 10000, 30000],
        "clf__C": [0.5, 1.0, 2.0]
    }
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=3, n_jobs=-1)
    gs.fit(Xtr, ytr)
    print("Best params:", gs.best_params_)

    best = gs.best_estimator_
    yhat = best.predict(Xte)
    acc = accuracy_score(yte, yhat)
    f1  = f1_score(yte, yhat)
    print("\n=== Improvement #3 (LinearSVC + TFIDF + GridSearch) ===")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(yte, yhat, digits=4))
    print("Confusion matrix:\n", confusion_matrix(yte, yhat))

    print("\n=== 5-Fold CV ===")
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for k, (tr, te) in enumerate(sk.split(X, y), 1):
        # dùng lại 'best' cho nhất quán pipeline/params
        best.fit([X[i] for i in tr], y[tr])
        yk = best.predict([X[i] for i in te])
        f1s.append(f1_score(y[te], yk))
        print(f"Fold {k}: F1={f1s[-1]:.4f}")
    print(f"F1 mean={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    main()
