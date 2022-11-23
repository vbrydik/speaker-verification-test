import numpy as np
from typing import Callable


class Pipeline:

    def __init__(
            self, 
            embedding_fn: Callable, 
            distance_fn: Callable,
        ):
        self.embedding_fn = embedding_fn
        self.distance_fn = distance_fn

    def __call__(
            self, 
            audio_1: np.ndarray, 
            audio_2: np.ndarray,
        ) -> float:
        emb_1 = self.embedding_fn(audio_1)
        emb_2 = self.embedding_fn(audio_2)
        return self.distance_fn(emb_1, emb_2)

