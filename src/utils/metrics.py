import torch


def metrics(predictions, targets):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)

    ranks = targets_sorted.argmax(dim=1) + 1
    mrr = torch.sum(1 / ranks)

    hits = []
    for k in [1, 3, 10]:
        hits_k = targets_sorted[:, :k].sum(dim=1).float()
        hits_k[hits_k > 1] = 1
        hits.append(hits_k.sum())

    return {
        "mrr" : mrr,
        "hits@1": hits[0],
        "hits@3": hits[1],
        "hits@10": hits[2],
    }
