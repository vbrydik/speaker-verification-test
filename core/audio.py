import soundfile as sf


class Audio:

    def __init__(self, path):
        _wav, _sr = sf.read(path)
        self.path = path
        self.wave = _wav
        self.sample_rate = _sr
    
    @property
    def mono(self):
        return self.wave.sum(axis=1) / 2
