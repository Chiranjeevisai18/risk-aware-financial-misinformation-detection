import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfidfLogistic:
    """TF-IDF vectorizer + Logistic Regression wrapper."""

    def __init__(self, cfg: dict):
        self.vectorizer = TfidfVectorizer(
            max_features=cfg["tfidf"]["max_features"],
            ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
        )
        self.clf = LogisticRegression(
            solver=cfg["logistic"]["solver"],
            max_iter=cfg["logistic"]["max_iter"],
            class_weight=cfg["logistic"]["class_weight"],
        )

    def fit(self, texts: list, labels: list):
        print("[Logistic] Fitting TF-IDF vectorizer and Logistic Regression …")
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        print("[Logistic] ✅ Training complete.")

    def predict_proba(self, texts: list) -> np.ndarray:
        """Returns shape (n, 4) probability matrix via predict_proba()."""
        X = self.vectorizer.transform(texts)
        probs = self.clf.predict_proba(X)
        return probs  # already sums to 1 by sklearn convention

    def save(self, export_dir: str):
        joblib.dump(self.vectorizer, f"{export_dir}/tfidf_vectorizer.pkl")
        joblib.dump(self.clf, f"{export_dir}/logistic_model.pkl")
        print(f"[Logistic] Artifacts saved to {export_dir}/")

    @classmethod
    def load(cls, export_dir: str) -> "TfidfLogistic":
        obj = cls.__new__(cls)
        obj.vectorizer = joblib.load(f"{export_dir}/tfidf_vectorizer.pkl")
        obj.clf = joblib.load(f"{export_dir}/logistic_model.pkl")
        return obj
