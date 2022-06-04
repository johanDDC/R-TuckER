import torch
from torch import nn, sparse_coo_tensor
from torch.nn.init import xavier_normal_, uniform_
from torch.autograd import Variable

import numpy as np
from opt_einsum import contract

from tucker_riemopt import Tucker
from tucker_riemopt import backend
from tucker_riemopt.riemopt import group_cores

from utils import reshape_fortran


class R_TuckER(torch.nn.Module):
    def __init__(self, data, rank, shape, device, **kwargs):
        super().__init__()

        self.E1 = torch.tensor(np.random.uniform(-1, 1, (shape[0], rank[0])),
                                                 dtype=torch.float32, device=device, requires_grad=True)
        # self.E1 = torch.nn.Parameter(self.E1)
        self.R = torch.tensor(np.random.uniform(-1, 1, (shape[1], rank[1])),
                                                 dtype=torch.float32, device=device, requires_grad=True)
        # self.R = torch.nn.Parameter(self.R)
        self.E2 = torch.tensor(np.random.uniform(-1, 1, (shape[2], rank[2])),
                                                 dtype=torch.float32, device=device, requires_grad=True)
        # self.E2 = torch.nn.Parameter(self.E2)
        self.W = torch.tensor(np.zeros(rank),
                                         dtype=torch.float, device=device, requires_grad=True)
        # self.W = torch.nn.Parameter(self.W)

        self.device = device
        self.rank = rank

        self.input_dropout = torch.nn.Dropout(0.1)
        self.hidden_dropout1 = torch.nn.Dropout(0.1)
        self.hidden_dropout2 = torch.nn.Dropout(0.1)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(rank[0])
        self.bn1 = torch.nn.BatchNorm1d(rank[2])

    def init(self):
        # pass
        xavier_normal_(self.E1.data)
        xavier_normal_(self.R.data)
        xavier_normal_(self.E2.data)

    def __g(self, core, factors, X):
        self.W = group_cores(self.W, core)
        self.E1 = torch.cat([self.E1, factors[0]], dim=1)
        self.R = torch.cat([self.R, factors[1]], dim=1)
        self.E2 = torch.cat([self.E2, factors[2]], dim=1)
        return self.forward(X)

    def __hosvd(self, A, eps=1e-6):
        def trunk(u, s, eps):
            eps_svd = eps / np.sqrt(3) * torch.sqrt(s @ s)
            cumsum = backend.cumsum(backend.flip(s))
            cumsum = backend.flip(~(cumsum <= eps_svd))
            return u[:, cumsum]

        width, heigth, length = A.shape
        U_hat = A.view(width, heigth * length)
        V_hat = reshape_fortran(A.permute((1, 0, 2)), (heigth, width * length))
        W_hat = reshape_fortran(A.permute((2, 0, 1)), (length, width * heigth))
        U, Su, _ = torch.linalg.svd(U_hat, full_matrices=False)
        V, Sv, _ = torch.linalg.svd(V_hat, full_matrices=False)
        W, Sw, _ = torch.linalg.svd(W_hat, full_matrices=False)
        U = trunk(U, Su, eps)
        V = trunk(V, Sv, eps)
        W = trunk(W, Sw, eps)
        G = contract("ijk,ia,jb,kc->abc", A, U, V, W)
        return G, U, V, W

    def __step(self, new_core, new_factors, lr):
        core = torch.zeros((self.W.shape[0] + new_core.shape[0],
                            self.W.shape[1] + new_core.shape[1],
                            self.W.shape[2] + new_core.shape[2]))
        core[:self.W.shape[0], :self.W.shape[1], :self.W.shape[2]] = self.W
        core[self.W.shape[0]:, self.W.shape[1]:, self.W.shape[2]:] = -lr * new_core
        factors = [torch.cat([self.E1, new_factors[0]], dim=1),
                   torch.cat([self.R,  new_factors[1]], dim=1),
                   torch.cat([self.E2, new_factors[2]], dim=1)]

        Q, R = [], []
        for factor in factors:
            q, r = torch.linalg.qr(factor)
            Q.append(q)
            R.append(r)

        G, U, V, W = self.__hosvd(contract("ijk,ai,bj,ck->abc", core, *R))

        self.W = G[:self.rank[0], :self.rank[1], :self.rank[2]].contiguous()
        self.E1 = Q[0] @ U[:, :self.rank[0]]
        self.R = Q[1] @ V[:, :self.rank[1]]
        self.E1 = Q[2] @ W[:, :self.rank[2]]


    def optimize(self, X, y, lr=1e-3):
        dS = torch.clone(self.W.detach())
        dU = [torch.clone(self.E1.detach()),
              torch.clone(self.R.detach()),
              torch.clone(self.E2.detach())]

        dS.requires_grad_(True)
        for factor in dU:
            factor.requires_grad_(True)

        predictions = self.__g(dS, dU, X)
        self.W = self.W[:self.rank[0], :self.rank[1], :self.rank[2]].detach()
        self.E1 = self.E1[:, :self.rank[0]].detach()
        self.R = self.R[:, :self.rank[1]].detach()
        self.E2 = self.E2[:, :self.rank[2]].detach()

        loss = self.loss(predictions, y)
        loss.backward()

        dS = dS.grad.detach()
        dU = [factor.grad.detach() for factor in dU]
        dU = [dU[0] - self.E1 @ (self.E1.T @ dU[0]),
              dU[1] - self.R @ (self.R.T @ dU[1]),
              dU[2] - self.E2 @ (self.E2.T @ dU[2])]

        new_core = group_cores(dS, self.W)
        new_factors = [torch.cat([self.E1, dU[0]], dim=1),
                       torch.cat([self.R, dU[1]], dim=1),
                       torch.cat([self.E2, dU[2]], dim=1)]
        self.__step(new_core, new_factors, lr)

    def forward(self, X):
        batch_size = X.shape[0]
        batch_arange = torch.arange(batch_size).to(self.device)
        subject_idx = torch.vstack([batch_arange, X[:, 0]])
        subject_idx = sparse_coo_tensor(subject_idx, torch.ones(subject_idx.shape[1]),
                                        (batch_size, self.E1.shape[0]), dtype=torch.float32,
                                        device=self.device)
        relation_idx = torch.vstack([batch_arange, X[:, 1]])
        relation_idx = sparse_coo_tensor(relation_idx, torch.ones(relation_idx.shape[1]),
                                         (batch_size, self.R.shape[0]), dtype=torch.float32,
                                         device=self.device)
        e1 = subject_idx @ self.E1
        # x = self.bn0(e1)
        x = self.input_dropout(e1)
        x = x.view(-1, 1, e1.size(1))

        r = relation_idx @ self.R
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        # x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E2.T)
        pred = torch.sigmoid(x)
        return pred