import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def prob_sanity_check(probs: np.ndarray, name: str):
    """
    Raise ValueError if `probs` is not a valid probability distribution.
    Checks:
      - sums to 1 (±1e-4 tolerance)
      - no negative values
      - no NaN values
    """
    if np.isnan(probs).any():
        raise ValueError(f"[{name}] NaN values detected in probabilities.")
    if (probs < 0).any():
        raise ValueError(f"[{name}] Negative probability detected.")
    row_sums = probs.sum(axis=1)
    if not np.isclose(row_sums, 1.0, atol=1e-4).all():
        raise ValueError(
            f"[{name}] Probabilities do not sum to 1. "
            f"Got row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )
