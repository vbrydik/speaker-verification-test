import os
import glob

import tqdm
import pandas as pd

from core.pipeline import Pipeline
from core.pyannote import Pyannote
from core.wavlm import WavLM
from core.titanet import TitaNet
from core.ecapa import Ecapa
from core.cosine_similarity import cosine_similarity
from core.metrics import compute_eer, compute_min_dcf
from utils.generate_speakers_dataset import (
    # generate_speakers_dataset,
    generate_speakers_file_pairs,
)


def get_label(file1: str, file2: str) -> int:
    """
    Return 0 if different speakers, 1 if same speakers.
    """
    return 0


def evaluate_pipeline(
    pipeline, 
    data, 
) -> pd.DataFrame:

    scores = []
    labels = []

    for file1, file2 in tqdm.tqdm(data, total=len(data)):
        similarity = pipeline(file1, file2)
        label = get_label(file1, file2)
        scores.append(similarity)
        labels.append(label)

    ee_rate, thresh, fa_rate, fr_rate = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(fr_rate, fa_rate)

    result = {
        "pipeline": pipeline.name,
        "fa_rate": fa_rate, 
        "fr_rate": fr_rate, 
        "ee_rate": ee_rate,
        "dcf": min_dcf, 
        "threshold": thresh,
    }
    return result


def main():
    # Get speaker to samples map
    dataset_dir = "./dataset"
    speakers = glob.glob(os.path.join(dataset_dir, "*"))
    # FIXME: Remove the line below before pushing
    speakers = speakers[:2]
    speaker_to_samples_dict = { s: glob.glob(os.path.join(s, "*.wav")) for s in speakers }

    # Generate datasets for evaluation for each speaker
    speakers_files_pairs = generate_speakers_file_pairs(speaker_to_samples_dict)

    total_df = pd.DataFrame()

    # Define pipelines
    pipelines = [
        Pipeline("pyannote", Pyannote(), cosine_similarity),
        Pipeline("wavlm-base", WavLM(device="cuda"), cosine_similarity),
        # Pipeline("wavlm-large", WavLM("microsoft/wavlm-large", device="cpu"), cosine_similarity),
        Pipeline("titanet", TitaNet(), cosine_similarity),
        Pipeline("ecapa", Ecapa(), cosine_similarity),
    ]

    for pipeline in pipelines:
        print(f"Evaluating pipeline: {pipeline.name}")
        pipe_scores_df = evaluate_pipeline(pipeline, speakers_files_pairs)
        pipe_scores_df.to_csv(f"{pipeline.name}_scores.csv", index=False)
        total_df = pd.concat([total_df, pipe_scores_df], axis=0, ignore_index=True)
    
    total_df.to_csv("all_scores.csv", index=False)


if __name__ == "__main__":
    main()
