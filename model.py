import torch
from torch import nn, sparse_coo_tensor
from torch.nn.init import xavier_normal_, uniform_
from torch.autograd import Variable

import numpy as np

from tucker_riemopt import Tucker
from tucker_riemopt import backend

class R_TuckER(torch.nn.Module):
    def __init__(self, shape, rank):
        """
        Parameters:
        -----------
        data_count: tuple
            (entities_count, relations_count) --- number of objects of corresponding kind in dataset
        embeddings_dim: tuple
            (entities_dim, relations_dim) --- dimensions of embeddings
        rank: Sequence[int]
            rank of the manifold
        """
        super(R_TuckER, self).__init__()
        if type(rank) is int:
            rank = [rank] * 3


        self.tucker = Tucker(backend.randn(rank, dtype=torch.float32),
                             [backend.randn((shape[i % 2], rank[i]), dtype=torch.float32) for i in range(len(rank))])
        self.device = "cpu"

        self.bn_0 = nn.BatchNorm1d(rank[0])
        self.bn_1 = nn.BatchNorm1d(rank[1])
        self.loss = nn.BCELoss(reduction="sum")
        # self.loss = nn.BCELoss()


    def init(self, rank):
        # xavier_normal_(self.tucker.core)
        # self.tucker = Tucker.full2tuck(self.tucker, rank)
        uniform_(self.tucker.core, -1, 1)
        # normal_(self.S)
        # normal_(self.R)
        # xavier_normal_(self.tucker.factors[0])
        # xavier_normal_(self.tucker.factors[1])
        # xavier_normal_(self.tucker.factors[2])
        # xavier_normal_(self.R)
        # xavier_normal_(self.core)

    def set_core(self, T):
        self.core = T

    def forward(self, subject_idx, relation_idx):
        batch_size = len(subject_idx)
        batch_arange = torch.arange(batch_size).to(self.device)
        subject_idx = torch.vstack([batch_arange, subject_idx])
        subject_idx = sparse_coo_tensor(subject_idx, torch.ones(subject_idx.shape[1]),
                                        (batch_size, self.tucker.factors[0].shape[0]), dtype=torch.float32, device=self.device)
        relation_idx = torch.vstack([batch_arange, relation_idx])
        relation_idx = sparse_coo_tensor(relation_idx, torch.ones(relation_idx.shape[1]),
                                        (batch_size, self.tucker.factors[1].shape[0]), dtype=torch.float32, device=self.device)
        pred = self.tucker.k_mode_product(0, subject_idx).k_mode_product(1, relation_idx)
        # pred.factors[0] = self.bn_0(pred.factors[0])
        # pred.factors[1] = self.bn_1(pred.factors[1])
        pred = pred.full()
        pred = torch.sigmoid(pred)
        preds = pred[0, 0, :].reshape(1, -1)
        for i in range(1, pred.shape[0]):
            preds = torch.cat([preds, pred[i, i, :].reshape(1, -1)], dim=0)

        def loss_fn(T, targets):
            pred = T.k_mode_product(0, subject_idx).k_mode_product(1, relation_idx)
            # pred.factors[0] = self.bn_0(pred.factors[0])
            # pred.factors[1] = self.bn_1(pred.factors[1])
            pred = pred.full()
            pred = torch.sigmoid(pred)
            preds = pred[0, 0, :].reshape(1, -1)
            for i in range(1, pred.shape[0]):
                preds = torch.cat([preds, pred[i, i, :].reshape(1, -1)], dim=0)
            return self.loss(preds, targets)

        return preds, loss_fn

    def to(self, device):
        if device == "cuda":
            self.cuda()
            self.device = "cuda"
        else:
            self.cpu()
            self.device = "cpu"
        self.tucker = Tucker(self.tucker.core.to(device),
                             [factor.to(device) for factor in self.tucker.factors])
