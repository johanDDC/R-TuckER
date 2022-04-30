import torch
from torch import nn
from torch.nn.init import xavier_normal_, xavier_uniform_

import numpy as np

from tucker_riemopt import Tucker
from tucker_riemopt import backend

class R_TuckER(torch.nn.Module):
    def __init__(self, data_count, embeddings_dim, rank=None, **kwargs):
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

        self.S = nn.Embedding(data_count[0], embeddings_dim[0])
        self.R = nn.Embedding(data_count[1], embeddings_dim[1])
        self.rank = rank
        self.core = backend.randn((embeddings_dim[0], embeddings_dim[1], embeddings_dim[0]), dtype=torch.float32)
        self.input_dropout = nn.Dropout(kwargs.get("input_dropout", 0))
        self.hidden_dropout1 = nn.Dropout(kwargs.get("hidden_dropout1", 0))
        self.hidden_dropout2 = nn.Dropout(kwargs.get("hidden_dropout2", 0))
        self.loss = nn.BCELoss()

        self.bn0 = nn.BatchNorm1d(embeddings_dim[0])
        self.bn1 = nn.BatchNorm1d(embeddings_dim[1])

    def init(self, rank):
        xavier_normal_(self.S.weight.data)
        xavier_normal_(self.R.weight.data)
        self.core = xavier_uniform_(self.core)
        self.core = Tucker.full2tuck(self.core, rank)

    def set_core(self, T):
        self.core = T

    def forward(self, subject_idx, relation_idx):
        subjects = self.S(subject_idx)
        relations = self.R(relation_idx)
        relations = self.bn1(relations)

        def forward_core(T, targets):
            x = self.bn0(subjects)
            W = T.k_mode_product(1, relations)
            W = W.k_mode_product(0, x)
            x = W.k_mode_product(2, self.S.weight)
            pred = torch.sigmoid(x.full())
            preds = pred[0, 0, :].reshape(1, -1)
            for i in range(1, pred.shape[0]):
                preds = torch.cat([preds, pred[i, i, :].reshape(1, -1)], dim=0)
            return self.loss(preds, targets)

        x = self.bn0(subjects)
        # x = self.input_dropout(x)
        # x.shape: batch_size x entities_dim
        W = self.core.k_mode_product(1, relations)
        W = W.k_mode_product(0, x)
        x = W.k_mode_product(2, self.S.weight)
        pred = torch.sigmoid(x.full())
        preds = pred[0, 0, :].reshape(1, -1)
        for i in range(1, pred.shape[0]):
            preds = torch.cat([preds, pred[i, i, :].reshape(1, -1)], dim=0)
        return preds, forward_core

    def to(self, device):
        if device == "cuda":
            self.cuda()
        else:
            self.cpu()
        self.set_core(Tucker(self.core.core.to(device), [factor.to(device) for factor in self.core.factors]))
