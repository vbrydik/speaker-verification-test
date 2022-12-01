import torch


_cosine_sim = torch.nn.CosineSimilarity(dim=-1)


def cosine_similarity(emb1, emb2):
    if not isinstance(emb1, torch.Tensor):
        emb1 = torch.tensor(emb1)
    if not isinstance(emb2, torch.Tensor):
        emb2 = torch.tensor(emb2)
    emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
    emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()
    return _cosine_sim(emb1, emb2).numpy()
