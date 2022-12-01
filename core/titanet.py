import nemo.collections.asr as nemo_asr

from core.audio import Audio


class TitaNet:

    def __init__(self):
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    
    def __call__(self, audio: Audio):
        return self.model.get_embedding(audio.path)
