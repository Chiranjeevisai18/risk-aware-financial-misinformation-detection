import numpy as np
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)

LABEL_NAMES = ["Safe", "Misleading", "High Risk", "Scam"]


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")


def per_class_f1(y_true, y_pred) -> np.ndarray:
    """Returns array of F1 per class [Safe, Misleading, High Risk, Scam]."""
    return f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])


def full_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        digits=4,
    )


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])


def print_confusion(y_true, y_pred):
    cm = confusion(y_true, y_pred)
    header = f"{'':15s}" + "".join(f"{n:>12s}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{LABEL_NAMES[i]:15s}" + "".join(f"{v:>12d}" for v in row)
        print(row_str)
