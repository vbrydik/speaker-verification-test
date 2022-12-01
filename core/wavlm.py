# from datasets import load_dataset
# import torch

# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
# model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

# # audio files are decoded on the fly
# audio = [x["array"] for x in dataset[:2]["audio"]]
# inputs = feature_extractor(audio, padding=True, return_tensors="pt")
# embeddings = model(**inputs).embeddings
# embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

# # the resulting embeddings can be used for cosine similarity-based retrieval
# cosine_sim = torch.nn.CosineSimilarity(dim=-1)
# similarity = cosine_sim(embeddings[0], embeddings[1])
# threshold = 0.86  # the optimal threshold is dataset-dependent
# if similarity < threshold:
#     print("Speakers are not the same!")

import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from core.audio import Audio



class WavLM:

    DEFAULT_MODEL_PATH = "microsoft/wavlm-base-plus-sv"

    def __init__(self, model_path: str = None, device: str = None):
        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device is not None:
            self.device = device

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = WavLMForXVector.from_pretrained(model_path).to(self.device)

    def __call__(self, audio1: Audio, audio2: Audio = None):
        use_batch = audio2 is not None

        if use_batch:
            inputs = self.feature_extractor(
                [audio1.mono, audio2.mono], 
                sampling_rate=16000, #audio.sample_rate, 
                padding=True, 
                return_tensors="pt",
            )
        else:
            inputs = self.feature_extractor(
                [audio1.mono], 
                sampling_rate=16000, #audio.sample_rate, 
                padding=True, 
                return_tensors="pt",
            )

        embeddings = self.model(**inputs.to(self.device)).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu().detach()

        if use_batch:
            return embeddings[0], embeddings[1]
        else:
            return embeddings[0]


if __name__ == "__main__":
    from core.cosine_similarity import cosine_similarity
    model = WavLM()

    audio1 = Audio("dataset/1-Zelenskyi/audio09.wav") 
    emb1 = model(audio1)

    audio2 = Audio("dataset/2-Sadovyi/audio01.wav")
    emb2 = model(audio2)

    print(cosine_similarity(emb1, emb2))



