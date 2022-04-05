import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_normal_

from tucker_riemopt import Tucker

class R_TuckER(torch.nn.Module):
    def __init__(self, dataset, entities_dim, relations_dim, rank = None, **kwargs):
        super(R_TuckER, self).__init__()

        self.S = nn.Embedding(len(dataset.entities), entities_dim)
        self.R = nn.Embedding(len(dataset.relations), relations_dim)
        self.core = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (entities_dim, relations_dim, entities_dim)),
                                    dtype=torch.float, requires_grad=True))
        self.core = Tucker.full2tuck(self.core)
        self.input_dropout = nn.Dropout(kwargs.get("input_dropout", 0))
        self.hidden_dropout1 = nn.Dropout(kwargs.get("hidden_dropout1", 0))
        self.hidden_dropout2 = nn.Dropout(kwargs.get("hidden_dropout2", 0))
        self.loss = nn.BCELoss()

        self.bn0 = nn.BatchNorm1d(entities_dim)
        self.bn1 = nn.BatchNorm1d(entities_dim)

    def init(self):
        xavier_normal_(self.S.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, subject_idx, relation_idx):
        subjects = self.S(subject_idx)
        x = self.bn0(subjects)
        x = self.input_dropout(x)
        # x.shape: batch_size x entities_dim

        relations = self.R(relation_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred
