import torch


def metrics(prediction, target):
    zero = torch.tensor([0], device=prediction.device)
    one = torch.tensor([1], device=prediction.device)
    target_col = torch.where(target == 1)[1].view(-1, 1)

    indices = prediction.argsort()
    mrr = (1.0 / (indices == target_col).nonzero()[:, 1].float().add(1.0)).sum().item()

    hits = []
    for k in [1, 3, 10]:
        hits.append(prediction.topk(k=k, largest=False)[1])
        hits[-1] = torch.where(hits[-1] == target_col, one, zero).float().sum().item()

    return {
        "mrr" : mrr,
        "hits@1": hits[0],
        "hits@3": hits[1],
        "hits@10": hits[2],
    }
