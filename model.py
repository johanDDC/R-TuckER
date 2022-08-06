import torch
from torch import nn
from torch.nn.init import xavier_normal_, xavier_uniform_

from tucker_riemopt import Tucker


class R_TuckER(torch.nn.Module):
    def __init__(self, data_count, rank=None, **kwargs):
        """
        Parameters:
        -----------
        data_count: tuple
            (entities_count, relations_count) --- number of objects of corresponding kind in dataset
        rank: tuple
            rank of the manifold
        rank: Sequence[int]
            rank of the manifold
        """
        super().__init__()

        self.knowledge_graph = Tucker(torch.zeros(tuple(rank), dtype=torch.float32),
                                      [torch.randn((data_count[1], rank[0]), dtype=torch.float32),
                                       torch.randn((data_count[0], rank[1]), dtype=torch.float32),
                                       torch.randn((data_count[0], rank[2]), dtype=torch.float32)])
        self.rank = rank
        self.device = "cpu"
        # self.loss = lambda preds, target: nn.BCELoss()(preds, target) + 1e-4 * self.knowledge_graph.norm() ** 2
        self.loss = nn.BCELoss()

        self.bn0 = nn.BatchNorm1d(rank[1])
        self.bn1 = nn.BatchNorm1d(rank[0])

    def init(self):
        xavier_uniform_(self.knowledge_graph.core)
        xavier_normal_(self.knowledge_graph.factors[0])
        xavier_normal_(self.knowledge_graph.factors[1])
        xavier_normal_(self.knowledge_graph.factors[2])

    def update_graph(self, T):
        self.knowledge_graph = T

    def forward(self, subject_idx, relation_idx):
        relations = self.knowledge_graph.factors[0][relation_idx, :]
        subjects = self.knowledge_graph.factors[1][subject_idx, :]

        def forward_core(T, targets):
            relations = T.factors[0][relation_idx, :]
            subjects = T.factors[1][subject_idx, :]
            preds = torch.einsum("abc,da->dbc", T.core, relations)
            preds = torch.bmm(subjects.view(-1, 1, subjects.shape[1]), preds).view(-1, subjects.shape[1])
            preds = preds @ T.factors[2].T
            preds = torch.sigmoid(preds)
            return self.loss(preds, targets)

        preds = torch.einsum("abc,da->dbc", self.knowledge_graph.core, relations)
        preds = torch.bmm(subjects.view(-1, 1, subjects.shape[1]), preds).view(-1, subjects.shape[1])
        preds = preds @ self.knowledge_graph.factors[2].T
        return torch.sigmoid(preds), forward_core

    def to(self, device):
        self.device = device
        if device == "cuda":
            self.cuda()
        else:
            self.cpu()
        self.update_graph(Tucker(self.knowledge_graph.core.to(device),
                                 [factor.to(device) for factor in self.knowledge_graph.factors]))
