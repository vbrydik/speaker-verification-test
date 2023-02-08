from typing import Callable
from core.pyannote import Audio
from core.cosine_similarity import cosine_similarity


class Pipeline:

    def __init__(
            self, 
            name: str,
            embedding_fn: Callable, 
            # score_fn: Callable,
        ):
        self.name = name
        self.embedding_fn = embedding_fn
        # self.score_fn = score_fn

    def __call__(
            self, 
            audio1: Audio or str, 
            audio2: Audio or str,
        ) -> float:
        if isinstance(audio1, str):
            audio1 = Audio(audio1)
        if isinstance(audio2, str):
            audio2 = Audio(audio2)
        emb1 = self.embedding_fn(audio1)
        emb2 = self.embedding_fn(audio2)
        # similarity = self.score_fn(emb1, emb2)
        similarity = cosine_similarity(emb1, emb2)
        return similarity


if __name__ == "__main__":
    from core.pyannote import Pyannote, cosine_similarity

    audio1 = Audio("dataset/1-Zelenskyi/audio01.wav")
    audio2 = Audio("dataset/2-Sadovyi/audio01.wav")

    pipeline = Pipeline(Pyannote())
    print(pipeline(audio1, audio2))
