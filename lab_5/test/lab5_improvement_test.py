# -*- coding: utf-8 -*-
"""
Improvement #1: TFIDF (1-2gram) + MultinomialNB
- In thêm: K-Fold, top n-grams theo log prob.
"""
import os, csv, random, numpy as np, re
from typing import List, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

random.seed(42); np.random.seed(42)

# ---- load data (giống baseline) ----
def _exists_large_csv():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "imdb_50k.csv"))
    return path if os.path.isfile(path) else None

def _read_csv(csv_path, limit=6000):
    texts, labels = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.lower() for c in reader.fieldnames]
        tcol = "text" if "text" in cols else "review"
        lcol = "label" if "label" in cols else "sentiment"
        for i, row in enumerate(reader):
            if limit and i >= limit: break
            txt = row.get(tcol) or row.get(tcol.title())
            lab = row.get(lcol) or row.get(lcol.title())
            if txt is None or lab is None: continue
            try: y = int(lab)
            except: y = 1 if str(lab).strip().lower() in {"pos","positive","1","true"} else 0
            texts.append(txt); labels.append(y)
    return texts, np.array(labels, int)

def _demo():
    base = [
        ("i love this movie so much",1),("fantastic acting and story",1),
        ("boring and too long",0),("bad acting and weak script",0),
    ]
    X, y = [], []
    for s,l in base:
        for v in ["","!"," very"," absolutely"," really"," quite"]:
            X.append((s+v).strip()); y.append(l)
    # nhân lên
    extra = []
    for s,l in list(zip(X,y)):
        extra.append((s[0].upper()+s[1:]+".", l))
    X2 = [s for s,_ in extra] + X
    y2 = [l for _,l in extra] + y
    return X2, np.array(y2, int)

def load_data():
    p = _exists_large_csv()
    if p: 
        X,y = _read_csv(p, limit=6000); print(f"== Using CSV: {p} | n={len(X)}")
    else:
        X,y = _demo(); print(f"== Using demo | n={len(X)}")
    return X,y

def evaluate(name, y, yhat, proba=None):
    print(f"\n=== {name} ===")
    acc = accuracy_score(y, yhat); f1 = f1_score(y, yhat)
    msg = f"Accuracy: {acc:.4f} | F1: {f1:.4f}"
    if proba is not None and len(np.unique(y))==2:
        try: msg += f" | ROC-AUC: {roc_auc_score(y, proba):.4f}"
        except: pass
    print(msg)
    print(classification_report(y, yhat, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y, yhat))

def top_features(pipe: Pipeline, k=15):
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: MultinomialNB = pipe.named_steps["clf"]
    feats = vec.get_feature_names_out()
    logp = clf.feature_log_prob_  # shape (2, V)
    top_pos = np.argsort(logp[1])[-k:][::-1]
    top_neg = np.argsort(logp[0])[-k:][::-1]
    print("\nTop POS n-grams:", ", ".join(feats[top_pos]))
    print("Top NEG n-grams:", ", ".join(feats[top_neg]))

def main():
    X,y = load_data()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=40000)),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    pipe.fit(Xtr,ytr)
    pr = None
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        pr = pipe.predict_proba(Xte)[:,1]
    yhat = pipe.predict(Xte)
    evaluate("Improvement #1 (TFIDF 1-2gram + MultinomialNB)", yte, yhat, pr)
    top_features(pipe, 20)

    print("\n=== 5-Fold CV ===")
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s=[]
    for k,(tr,te) in enumerate(sk.split(X,y),1):
        pipe.fit([X[i] for i in tr], y[tr])
        yk = pipe.predict([X[i] for i in te])
        f1s.append(f1_score(y[te], yk))
        print(f"Fold {k}: F1={f1s[-1]:.4f}")
    print(f"F1 mean={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__=="__main__":
    main()
