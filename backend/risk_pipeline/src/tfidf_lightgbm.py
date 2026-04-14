import joblib
import numpy as np
import lightgbm
from pathlib import Path


class TfidfLightGBM:
    """TF-IDF vectorizer + LightGBM wrapper for loading tuned artifacts."""

    def __init__(self):
        self.vectorizer = None
        self.clf = None

    def predict_proba(self, texts: list) -> np.ndarray:
        """Returns shape (n, 4) probability matrix via predict_proba()."""
        X = self.vectorizer.transform(texts)
        probs = self.clf.predict_proba(X)
        return probs

    def save(self, export_dir: str):
        # We re-save under the common exports directory so infer_hybrid can find it easily
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, f"{export_dir}/tfidf_vectorizer_lgb.pkl")
        joblib.dump(self.clf, f"{export_dir}/lgb_tuned_model.pkl")
        print(f"[LightGBM] Artifacts copied to {export_dir}/")

    @classmethod
    def load(cls, export_dir: str) -> "TfidfLightGBM":
        obj = cls.__new__(cls)
        obj.vectorizer = joblib.load(f"{export_dir}/tfidf_vectorizer_lgb.pkl")
        obj.clf = joblib.load(f"{export_dir}/lgb_tuned_model.pkl")
        return obj
