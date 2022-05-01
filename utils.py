import torch
from torch.optim import Optimizer, Adam
from tucker_riemopt import Tucker

from tucker_riemopt.optimize import get_line_search_tool
from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport
from torch.autograd import Variable


class R_TuckEROptimizer(Optimizer):
    def __init__(self, params, model, rank, lr, line_search_options=None):
        self.line_search = get_line_search_tool(line_search_options)
        self.rank = rank
        self.model = model
        defaults = dict(model=model, rank=rank, lr=lr, line_search=self.line_search)
        super().__init__(params, defaults)
        self.alpha = lr
        self.lr = None
        if lr == "Optimal":
            self.lr = True
            self.line_search = get_line_search_tool(line_search_options)
            self.alpha = self.line_search.alpha_0


    def calc_loss(self, predictions, targets):
        loss = self.model.loss(predictions, targets)
        return loss

    def fit(self, loss_fn, targets):
        x_k = self.model.tucker
        func = lambda T: loss_fn(T, targets)
        self.riemann_grad = compute_gradient_projection(func, x_k)
        if self.lr:
            self.alpha = self.line_search.line_search(func, x_k, self.riemann_grad, -self.riemann_grad,
                                                      self.rank, 1)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        x_k = self.model.tucker
        x_k -= self.alpha * self.riemann_grad
        x_k = x_k.round(self.rank)
        x_k.factors[2] = x_k.factors[0]
        self.model.tucker = x_k
        del self.riemann_grad


class RSVRG(Optimizer):
    def __init__(self, params, model, rank, lr, dataloader, memory=10, line_search_options=None):
        self.line_search = get_line_search_tool(line_search_options)
        self.rank = rank
        self.model = model
        self.lr = lr
        self.memory = memory
        self.dataloader = dataloader
        defaults = dict(model=model, rank=rank, lr=lr, dataloader=dataloader, memory=memory, line_search=self.line_search)
        super().__init__(params, defaults)
        self.rieman_grad = None
        self.idx = torch.randint(0, len(dataloader), (self.memory,))
        self.idx, _ = torch.sort(self.idx)
        self.fit_count = 0
        self.saved_grads = []
        self.batches = []

    def loss(self, predictions, targets):
        loss = self.model.loss(predictions, targets)
        return Variable(loss, requires_grad=True)

    def fit(self, loss_fn, targets):
        func = lambda T: loss_fn(T, targets)
        x_k = self.model.tucker
        if self.rieman_grad is None:
            self.fit_count = 0
            self.saved_grads = []
            self.rieman_grad = compute_gradient_projection(func, x_k)
            self.rieman_grad_rank = self.rieman_grad.rank
            if 0 in self.idx:
                self.saved_grads.append(self.rieman_grad)
        else:
            self.fit_count += 1
            grad = compute_gradient_projection(func, x_k)
            self.rieman_grad += grad
            self.rieman_grad = self.rieman_grad.round(self.rieman_grad_rank)
            if self.fit_count in self.idx:
                self.saved_grads.append(grad)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        x_k = self.model.tucker
        self.rieman_grad = (1 / self.fit_count) * self.rieman_grad
        x_opt = self.model.tucker
        for t in range(self.memory):
            # if t == 0:
            #     v = self.saved_grads[t] - vector_transport(x_opt, x_k, self.saved_grads[t] - self.rieman_grad)
            # else:
            v = self._restore_grad(t, x_k) - vector_transport(x_opt, x_k, self.saved_grads[t] - self.rieman_grad)
            x_k -= self.lr * v
            x_k = x_k.round(self.rank)
            del self.saved_grads[t]
        x_k.factors[2] = x_k.factors[0]
        self.model.tucker = x_k
        del self.rieman_grad
        self.rieman_grad = None

    def _grad(self, func, T):
        def pad(tensor, pad_width, constant_values):
            from torch.nn.functional import pad
            flat_pad_width = []
            for pair in pad_width:
                flat_pad_width.append(pair[1])
                flat_pad_width.append(pair[0])
            flat_pad_width = flat_pad_width[::-1]
            return pad(tensor, flat_pad_width, "constant", 0)

        def group_cores(core1, core2):
            d = len(core1.shape)
            r = core1.shape

            new_core = core1
            to_concat = core2

            for i in range(d):
                to_concat = pad(to_concat, [(0, r[j]) if j == i - 1 else (0, 0) for j in range(d)],
                                     constant_values=0)
                new_core = torch.cat([new_core, to_concat], axis=i)

            return new_core

        def g(T1, core, factors):
            new_factors = [torch.cat([T1.factors[i], factors[i]], axis=1) for i in range(T1.ndim)]
            new_core = group_cores(core, T1.core)

            T = Tucker(new_core, new_factors)
            return func(T)

        fs = [torch.zeros_like(T.factors[i]) for i in range(T.ndim)]
        torch.autograd.backward(g(T, T.core, fs))
        dS = T.core.grad
        dU = [fs[i].grad for i in range(3)]
        dU = [dU[i] - T.factors[i] @ (T.factors[i].T @ dU[i]) for i in range(len(dU))]
        return Tucker(group_cores(dS, T.core), [torch.cat([T.factors[i], dU[i]], axis=1) for i in range(T.ndim)])

    def _restore_grad(self, idx, T):
        if len(self.batches) == 0:
            for i, batch in enumerate(self.dataloader):
                if i in self.idx:
                    self.batches.append(batch)
                if len(self.batches) == len(self.idx):
                    break
        features, targets = self.batches[idx]
        features = features.to(self.model.device)
        targets = targets.to(self.model.device).float()
        predictions, loss_fn = self.model(features[:, 0], features[:, 1])
        loss = self.loss(predictions, targets)
        func = lambda T: loss_fn(T, targets)
        return self._grad(func, T)




def MRR_metrics(predictions, targets):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    ranks = targets_sorted.argmax(1) + 1
    mrr = torch.mean(1 / ranks)
    return mrr


def hits_k_metrics(predictions, targets, k = 1):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    hits = targets_sorted[:, :k].sum(1).float()
    hits[hits > 1] = 1
    return torch.mean(hits)


def compute_metrics(predictions, targets, ks, accum=None):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    ranks = targets_sorted.argmax(1) + 1
    metrics = dict(mrr=0) if not accum else accum
    metrics["mrr"] += torch.sum(1 / ranks)
    for k in ks:
        if not accum:
            metrics[f"hits_{k}"] = 0
        hits = targets_sorted[:, :k].sum(1).float()
        hits[hits > 1] = 1
        metrics["hits_" + str(k)] += torch.sum(hits)
    return metrics


def filter_predictions(predictions, targets, filter):
    # removing all scores of actual true triplets in predictions but one we interested in.
    # useful for computing filtered MRR and Hits@k
    interest_prediction_vals = predictions.gather(1, filter)
    predictions[targets == 1] = 0
    targets[targets == 1] = 0
    return predictions.scatter_(1, filter, interest_prediction_vals), \
            targets.scatter_(1, filter, torch.ones(interest_prediction_vals.shape, device=targets.device))