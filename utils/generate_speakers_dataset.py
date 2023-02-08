import random
import itertools
from typing import List, Tuple


def one_speaker_pairs(x: list):
    pairs = []
    for pair in itertools.permutations(x, r=2):
        if pair[::-1] not in pairs:
            pairs.append(pair)
    return pairs



def generate_speakers_dataset(speaker_to_samples_dict: dict):
    speaker_to_subset_dict = { k: [] for k in speaker_to_samples_dict.keys() }

    for k, v in speaker_to_samples_dict.items():
        # Get permutations of the same speaker files
        same_files = one_speaker_pairs(v)
        # Get random files from other speakers
        other_files = []
        for _k, _v in speaker_to_samples_dict.items():
            if _k == k:
                continue
            other_files += _v
        
        _target = random.choices(v, k=len(same_files))
        _other = random.choices(other_files, k=len(same_files))
        other_files = list(zip(_target, _other))
        # Create subset for a speaker
        for f in same_files:
            speaker_to_subset_dict[k].append((f, 1))
        for f in other_files:
            speaker_to_subset_dict[k].append((f, 0))

    return speaker_to_subset_dict


def generate_speakers_file_pairs(speaker_to_samples_dict: dict) -> List[Tuple[str, str]]:
    pairs = []

    for k, v in speaker_to_samples_dict.items():
        
        # Get pairs of same speakers
        same_speaker_pairs = one_speaker_pairs(v)

        # Get pairs of different speakers, but the same number as the same speakers
        diff_speaker_pairs = []
        for _k, _v in speaker_to_samples_dict.items():
            if _k == k:
                continue
            diff_speaker_pairs += _v
        
        _target = random.choices(v, k=len(same_speaker_pairs))
        _other = random.choices(diff_speaker_pairs, k=len(same_speaker_pairs))
        diff_speaker_pairs = list(zip(_target, _other))

        # Extend pairs
        pairs.extend(same_speaker_pairs)
        pairs.extend(diff_speaker_pairs)

    return pairs
