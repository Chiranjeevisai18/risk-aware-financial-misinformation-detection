import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_validate(csv_path: str) -> pd.DataFrame:
    """
    Load dataset and enforce strict quality checks:
      - No null values
      - At least 8000 samples
      - Roughly balanced classes (~2000 each, ±200 tolerance)
    Raises ValueError and halts the pipeline if any check fails.
    """
    print(f"[Data] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # ── Null check ────────────────────────────────────────────────────────────
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"[Data] Dataset contains null values in columns: {null_cols}")

    # ── Minimum samples ───────────────────────────────────────────────────────
    if len(df) < 8000:
        raise ValueError(
            f"[Data] Total samples ({len(df)}) is below the required minimum of 8,000."
        )

    # ── Class balance ─────────────────────────────────────────────────────────
    class_counts = df["label"].value_counts().sort_index()
    n_classes = df["label"].nunique()
    expected_per_class = len(df) // n_classes
    tolerance = expected_per_class * 0.20  # 20% tolerance
    
    print(f"[Data] Class distribution:\n{class_counts.to_string()}")
    imbalanced = [lbl for lbl, cnt in class_counts.items() if abs(cnt - expected_per_class) > tolerance]
    if imbalanced:
        raise ValueError(
            f"[Data] Classes are not balanced (tolerance ±{tolerance} around {expected_per_class}). "
            f"Imbalanced labels: {imbalanced}. Counts:\n{class_counts.to_dict()}"
        )

    # ── Required columns ──────────────────────────────────────────────────────
    for col in ("text", "label"):
        if col not in df.columns:
            raise ValueError(f"[Data] Required column '{col}' not found in dataset.")

    print(f"[Data] ✅ Validation passed — {len(df)} samples, {df['label'].nunique()} classes.")
    return df


def stratified_split(df: pd.DataFrame, seed: int = 42):
    """
    70% train / 15% val / 15% test — stratified by label.
    Returns (train_df, val_df, test_df).
    """
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=seed
    )
    print(
        f"[Split] train={len(train)}, val={len(val)}, test={len(test)}"
    )
    return train, val, test
