import io
import torch
import soundfile as sf

from pyannote.audio import Model, Inference

from core.audio import Audio
from core.cosine_similarity import cosine_similarity


def _get_model():
    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token="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"
    )
    inference = Inference(model, window="whole")
    return inference


class Pyannote:

    def __init__(self):
        self.model = _get_model()

    def __call__(self, audio: Audio):
        return self.model(audio.path)


if __name__ == "__main__":
    model = Pyannote()

    audio1 = Audio("dataset/1-Zelenskyi/audio09.wav") 
    emb1 = model(audio1)

    audio2 = Audio("dataset/2-Sadovyi/audio01.wav")
    emb2 = model(audio2)

    print(cosine_similarity(emb1, emb2))

