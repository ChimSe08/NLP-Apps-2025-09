"""
TextClassifier: trains Logistic Regression for text classification.
Implements fit / predict / evaluate using scikit-learn.
"""
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextClassifier:
    def __init__(self, vectorizer):
        """
        vectorizer must implement fit_transform(list[str]) and transform(list[str])
        """
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self._model = LogisticRegression(solver="liblinear", random_state=42)
        self._model.fit(X, labels)
        return self

    def predict(self, texts: List[str]) -> List[int]:
        if self._model is None:
            raise RuntimeError("Model is not trained. Call fit(...) first.")
        X = self.vectorizer.transform(texts)
        return self._model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
