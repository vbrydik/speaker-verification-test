from typing import Any

from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim


def cosine_similarity(x: Any, y: Any) -> float:
    return _cosine_sim(x, y).mean()
