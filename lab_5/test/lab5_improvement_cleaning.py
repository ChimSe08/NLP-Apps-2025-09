# -*- coding: utf-8 -*-
"""
Improvement #2:
- Cleaning (lower, remove html, urls, punctuation, digits)
- TFIDF (1-2gram) + LogisticRegression
- GridSearch một vài tham số + báo cáo KFold chi tiết
"""
import os, csv, re, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _csv():
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "imdb_50k.csv"))
    return p if os.path.isfile(p) else None

def _read(csv_path, limit=6000):
    X,y=[],[]
    with open(csv_path,encoding="utf-8") as f:
        r=csv.DictReader(f); cols=[c.lower() for c in r.fieldnames]
        tcol="text" if "text" in cols else "review"
        lcol="label" if "label" in cols else "sentiment"
        for i,row in enumerate(r):
            if i==limit: break
            txt=row.get(tcol) or row.get(tcol.title()); lab=row.get(lcol) or row.get(lcol.title())
            if txt is None or lab is None: continue
            try: lab=int(lab)
            except: lab=1 if str(lab).lower() in {"pos","positive","1","true"} else 0
            X.append(clean_text(txt)); y.append(lab)
    return X, np.array(y,int)

def _demo():
    toy = [
        ("This movie is fantastic!! Visually stunning and well-acted.",1),
        ("Awful. Boring and way too long...",0),
        ("I loved the characters, the story was engaging.",1),
        ("Cheap effects, bad dialogue, I regret watching.",0),
    ]
    X=[clean_text(s) for s,_ in toy]*25
    y=np.array([l for _,l in toy]*25, int)
    return X,y

def load_data():
    p=_csv()
    if p:
        X,y=_read(p); print(f"== Using CSV cleaned | n={len(X)}")
    else:
        X,y=_demo(); print(f"== Using cleaned demo | n={len(X)}")
    return X,y

def main():
    X,y=load_data()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    grid = {
        "tfidf__max_features": [2000, 10000, 30000],
        "clf__C": [0.5, 1.0, 2.0]
    }
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=3, n_jobs=-1)
    gs.fit(Xtr, ytr)
    print("Best params:", gs.best_params_)

    best: Pipeline = gs.best_estimator_
    yhat = best.predict(Xte)
    proba = None
    if hasattr(best.named_steps["clf"], "predict_proba"):
        proba = best.predict_proba(Xte)[:,1]

    acc = accuracy_score(yte,yhat); f1=f1_score(yte,yhat)
    msg=f"Accuracy: {acc:.4f} | F1: {f1:.4f}"
    if proba is not None:
        try: msg+=f" | ROC-AUC: {roc_auc_score(yte, proba):.4f}"
        except: pass
    print("\n=== Improvement #2 (Cleaning + LogReg + GridSearch) ===")
    print(msg)
    print(classification_report(yte,yhat,digits=4))
    print("Confusion matrix:\n", confusion_matrix(yte,yhat))

    # KFold Report
    print("\n=== 5-Fold report ===")
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s=[]
    for k,(tr,te) in enumerate(sk.split(X,y),1):
        best.fit([X[i] for i in tr], y[tr])
        yk = best.predict([X[i] for i in te])
        f1s.append(f1_score(y[te], yk))
        print(f"Fold {k}: F1={f1s[-1]:.4f}")
    print(f"F1 mean={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__=="__main__":
    main()
