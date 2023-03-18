import tqdm
import pandas as pd

from core import (
    Pipeline,
    Pyannote,
    WavLM,
    TitaNet,
    Ecapa
)
from core.metrics import (
    compute_eer, 
    compute_min_dcf,
    compute_far_frr,
)
from utils import make_dataset


def get_label(file1: str, file2: str) -> int:
    """
    Return 0 if different speakers, 1 if same speakers.
    """
    def _get_name(x):
        return x.split("/")[-2]

    return int(_get_name(file1) == _get_name(file2))


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
    fa_score, fr_score = compute_far_frr(scores, labels, thresh)

    result = {
        "pipeline": pipeline.name,
        "fa_score": fa_score,
        "fr_score": fr_score,
        "ee_rate": ee_rate,
        "dcf": min_dcf, 
        "threshold": thresh,
    }
    return result


def main():

    dataset = make_dataset("./dataset")
    print(f"Number of pairs in dataset: {len(dataset)}")

    # Define pipelines
    pipelines = [
        Pipeline("pyannote", Pyannote()),
        Pipeline("wavlm-base", WavLM(device="cuda")),
        Pipeline("titanet", TitaNet()),
        Pipeline("ecapa", Ecapa()),
        # Pipeline("wavlm-large", WavLM("microsoft/wavlm-large", device="cpu")),
    ]

    results = {}

    for pipeline in pipelines:
        print(f"Evaluating pipeline: {pipeline.name}")
        results[pipeline.name] = evaluate_pipeline(pipeline, dataset)
    
    # Store resutls in a csv file
    pd.DataFrame(results).transpose().to_csv("scores.csv")

    print("Done!")


if __name__ == "__main__":
    main()
