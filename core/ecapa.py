import torchaudio
from speechbrain.pretrained import EncoderClassifier

from core.audio import Audio


class Ecapa:

    def __init__(self):
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    def __call__(self, audio: Audio):
        signal, fs = torchaudio.load(audio.path)
        return self.model.encode_batch(signal)