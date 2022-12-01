import os
import glob

import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from core.audio import Audio
from core.pipeline import Pipeline
from core.pyannote import Pyannote
from core.wavlm import WavLM
from core.titanet import TitaNet
from core.ecapa import Ecapa
from core.cosine_similarity import cosine_similarity
from utils.generate_speakers_dataset import generate_speakers_dataset


def far_frr_scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    return far, frr


def evaluate_pipeline(
    pipeline, 
    speakers_datasets, 
    threshold: float = 0.5
) -> pd.DataFrame:

    # Create table
    scores_df = pd.DataFrame(columns=["pipeline", "speaker", "far", "frr", "accuracy", "threshold"])

    for speaker, data in tqdm.tqdm(speakers_datasets.items(), total=len(speakers_datasets)):
        # Make predictions
        y_true = [label for _, label in data]
        y_pred = []
        for pair, _ in tqdm.tqdm(data, total=len(data)):
            similarity = pipeline(*(Audio(a) for a in pair))
            y_pred.append(int(similarity > threshold))
        # Calculate metrics
        far_score, frr_score = far_frr_scores(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        # Update table
        row = {
            "pipeline": pipeline.name,
            "speaker": speaker, 
            "far": far_score, 
            "frr": frr_score, 
            "accuracy": accuracy, 
            "threshold": threshold,
        }
        scores_df = scores_df.append(row, ignore_index=True)
    
    return scores_df


def main():
    # Get speaker to samples map
    dataset_dir = "./dataset"
    speakers = glob.glob(os.path.join(dataset_dir, "*"))
    speakers = speakers[:2]
    speaker_to_samples_dict = { s: glob.glob(os.path.join(s, "*.wav")) for s in speakers }

    # Generate datasets for evaluation for each speaker
    speakers_datasets = generate_speakers_dataset(speaker_to_samples_dict)

    total_df = pd.DataFrame()

    # Define pipelines
    pipelines = [
        Pipeline("pyannote", Pyannote(), cosine_similarity),
        Pipeline("wavlm-base", WavLM(device="cuda"), cosine_similarity),
        # Pipeline("wavlm-large", WavLM("microsoft/wavlm-large", device="cpu"), cosine_similarity, batch=True),
        Pipeline("titanet", TitaNet(), cosine_similarity),
        Pipeline("ecapa", Ecapa(), cosine_similarity),
    ]
    threshold = 0.5

    for pipeline in pipelines:
        print(f"Evaluating pipeline: {pipeline.name}")
        pipe_scores_df = evaluate_pipeline(pipeline, speakers_datasets, threshold)
        pipe_scores_df.to_csv(f"{pipeline.name}_scores.csv", index=False)

        total_df = pd.concat([total_df, pipe_scores_df], axis=0, ignore_index=True)
    
    total_df.to_csv("all_scores.csv", index=False)


if __name__ == "__main__":
    main()
