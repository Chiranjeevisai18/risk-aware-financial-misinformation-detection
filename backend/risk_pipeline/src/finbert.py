import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score


class FinBertDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class FinBertModel:
    """FinBERT fine-tuning and inference wrapper."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        model_name = cfg["finbert"]["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4, ignore_mismatched_sizes=True, attn_implementation="eager"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[FinBERT] Running on: {self.device}")
        self.is_stable = True  # flipped to False on over-fitting

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, train_df, val_df):
        export_dir = self.cfg["export_dir"]
        fb_cfg = self.cfg["finbert"]
        crit = self.cfg["criteria"]

        train_set = FinBertDataset(
            train_df["text"].tolist(), train_df["label"].tolist(),
            self.tokenizer, fb_cfg["max_length"]
        )
        val_set = FinBertDataset(
            val_df["text"].tolist(), val_df["label"].tolist(),
            self.tokenizer, fb_cfg["max_length"]
        )
        train_loader = DataLoader(train_set, batch_size=fb_cfg["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=fb_cfg["batch_size"])

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=fb_cfg["learning_rate"]
        )
        total_steps = len(train_loader) * fb_cfg["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        best_val_f1   = -np.inf
        patience      = fb_cfg["early_stop_patience"]
        no_improve    = 0
        overfit_gap   = crit["finbert_overfit_gap"]
        checkpoint    = os.path.join(export_dir, "finbert_best")

        for epoch in range(fb_cfg["epochs"]):
            # ── Train ──────────────────────────────────────────────────
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # ── Compute F1 on both sets ────────────────────────────────
            train_f1 = self._compute_f1(train_loader)
            val_f1   = self._compute_f1(val_loader)
            gap      = train_f1 - val_f1

            print(
                f"[FinBERT] Epoch {epoch+1}/{fb_cfg['epochs']} | "
                f"loss={avg_loss:.4f} | train_f1={train_f1:.4f} | "
                f"val_f1={val_f1:.4f} | gap={gap:.4f}"
            )

            # ── Over-fitting detection ─────────────────────────────────
            if gap > overfit_gap:
                print(
                    f"[FinBERT] ⚠️  Over-fit detected (gap={gap:.4f} > {overfit_gap}) "
                    f"— stopping early."
                )
                self.is_stable = False
                break

            # ── Early stopping on val_f1 ───────────────────────────────
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve  = 0
                self.model.save_pretrained(checkpoint)
                self.tokenizer.save_pretrained(checkpoint)
                print(f"[FinBERT] ✅ Best checkpoint saved (val_f1={val_f1:.4f}).")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("[FinBERT] Early stopping — no validation improvement.")
                    break

        # Always load the best checkpoint for inference
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"[FinBERT] Best checkpoint loaded from {checkpoint}.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, texts: list) -> np.ndarray:
        """Returns shape (n, 4) softmax probabilities."""
        self.model.eval()
        all_probs = []
        batch_size = self.cfg["finbert"]["batch_size"]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.cfg["finbert"]["max_length"],
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_f1(self, loader: DataLoader) -> float:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                preds  = logits.argmax(dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        return f1_score(all_labels, all_preds, average="macro")

    def save(self, export_dir: str):
        # Checkpoint already saved during training as finbert_best
        pass

    @classmethod
    def load(cls, cfg: dict) -> "FinBertModel":
        checkpoint = os.path.join(cfg["export_dir"], "finbert_best")
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        obj.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        obj.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj.model.to(obj.device)
        obj.is_stable = True
        return obj
