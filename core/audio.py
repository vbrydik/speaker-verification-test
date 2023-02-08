import soundfile as sf


class Audio:

    def __init__(self, path: str, sample_rate: int = 16000):
        _wav, _sr = sf.read(path)

        if sample_rate != _sr:
            print(
                "Warning! The file sample rate does not"
                f"match the wanted sample rate {sample_rate} != {_sr}"
            )
            # TODO: Resample audio here if sample rates do not match.

        self.path = path
        self.wave = _wav
        self.sample_rate = _sr
    
    @property
    def mono(self):
        return self.wave.sum(axis=1) / 2
