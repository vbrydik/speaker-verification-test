import numpy as np
from typing import Callable
from core.pyannote import Audio


class Pipeline:

    def __init__(
            self, 
            name: str,
            embedding_fn: Callable, 
            similarity_fn: Callable,
            batch: bool = False,
        ):
        self.name = name
        self.batch = batch
        self.embedding_fn = embedding_fn
        self.similarity_fn = similarity_fn

    def __call__(
            self, 
            audio1: Audio, 
            audio2: Audio,
        ) -> float:
        if self.batch:
            emb1, emb2 = self.embedding_fn(audio1, audio2)
        else:
            emb1 = self.embedding_fn(audio1)
            emb2 = self.embedding_fn(audio2)
        similarity = self.similarity_fn(emb1, emb2)
        if isinstance(similarity, np.ndarray):
            similarity = similarity.mean()
        return similarity


if __name__ == "__main__":
    from core.pyannote import Pyannote, cosine_similarity

    audio1 = Audio("dataset/1-Zelenskyi/audio01.wav")
    audio2 = Audio("dataset/2-Sadovyi/audio01.wav")

    pipeline = Pipeline(Pyannote(), cosine_similarity)
    print(pipeline(audio1, audio2))
