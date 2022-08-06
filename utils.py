import torch
from torch.optim import Optimizer, Adam

from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport
import numpy as np


class R_TuckEROptimizer(Optimizer):
    def __init__(self, params, model, rank, lr=None, adam_lr=3e-3,
                 scheduler_constructor=None, momentum_beta=0.9):
        self.rank = rank
        self.model = model
        self.lr = lr
        self.adam_lr = adam_lr
        self.momentum_beta = momentum_beta
        defaults = dict(model=model, rank=rank, lr=self.lr, adam_lr=self.adam_lr,
                        momentum_beta=self.momentum_beta)
        super().__init__(params, defaults)
        self.regular_optim = Adam(model.parameters(), lr=self.adam_lr)
        self.momentum = None
        self.direction = None
        if scheduler_constructor:
            self.scheduler = scheduler_constructor(self.regular_optim)

    def loss(self, predictions, targets):
        return self.model.loss(predictions, targets)

    def fit(self, loss_fn, targets):
        x_k = self.model.knowledge_graph
        if self.direction:
            self.momentum = vector_transport(None, x_k, self.direction)
        func = lambda T: loss_fn(T, targets)
        riemann_grad = compute_gradient_projection(func, x_k)
        grad_norm = riemann_grad.norm()
        riemann_grad = 1 / grad_norm * riemann_grad
        self.direction = -self.lr * riemann_grad if self.momentum is None else \
            -self.lr * riemann_grad + self.momentum_beta * self.momentum
        return grad_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        x_k = self.model.knowledge_graph
        x_k += self.direction
        x_k = x_k.round(self.rank)
        self.model.update_graph(x_k)
        self.regular_optim.step()

    def scheduler_step(self):
        self.scheduler.step()


def MRR_metrics(predictions, targets):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    ranks = targets_sorted.argmax(1) + 1
    mrr = torch.mean(1 / ranks)
    return mrr


def hits_k_metrics(predictions, targets, k=1):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    hits = targets_sorted[:, :k].sum(1).float()
    hits[hits > 1] = 1
    return torch.mean(hits)


def filter_predictions(predictions, targets, filter):
    # removing all scores of actual true triplets in predictions but one we interested in.
    # useful for computing filtered MRR and Hits@k
    interest_prediction_vals = predictions.gather(1, filter)
    predictions[targets == 1] = 0
    targets[targets == 1] = 0
    return predictions.scatter_(1, filter, interest_prediction_vals), \
           targets.scatter_(1, filter, torch.ones(interest_prediction_vals.shape, device=targets.device))


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def update_metrics(metrics, local_metrics):
    for key in metrics.keys():
        metrics[key].append(local_metrics[key])


def save_model(model, file_name, losses, metrics, epoch):
    meta = {
        "losses": losses,
        "metrics": metrics
    }
    torch.save(meta, "meta.pt")
    torch.save(model, f"{file_name}.pt")


def load_model(file_name, last_epoch=None):
    losses = None
    metrics = None
    try:
        meta = torch.load("meta.pt")
        losses = meta["losses"]
        metrics = meta["metrics"]
        if last_epoch:
            losses = {key: losses[key][:last_epoch] for key in losses.keys()}
            metrics = {key: metrics[key][:last_epoch] for key in metrics.keys()}
    except:
        pass
    model = torch.load(f"{file_name}.pt")
    return (losses, metrics), model


def draw_plots(losses, metrics, baselines=None):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    f, ax = plt.subplots(ncols=3, nrows=2, figsize=(24, 9))
    ax[0, 0].set(title="NLL loss", xlabel="Epoches", ylabel="Loss")
    ax[0, 1].set(title="Mean grad norm on train", xlabel="Epoches", ylabel="Norm")
    ax[0, 2].set(title="MRR", xlabel="Epoches", ylabel="MRR")
    ax[1, 0].set(title="Hits@1", xlabel="Epoches", ylabel="Hits@1")
    ax[1, 1].set(title="Hits@3", xlabel="Epoches", ylabel="Hits@3")
    ax[1, 2].set(title="Hits@10", xlabel="Epoches", ylabel="Hits@10")

    epoch = len(metrics["mrr"])
    x = np.arange(1, epoch + 1)
    baselines_y = np.ones(len(x))
    # if len(x) > 1:
    ax[0, 0].plot(x, losses["train"], c="tab:blue", label="train")
    ax[0, 0].plot(x, losses["val"], c="tab:green", label="val")
    ax[0, 0].plot(x, losses["test"], c="tab:orange", label="test")

    ax[0, 1].plot(x, losses["norm"], c="tab:red")
    ax[0, 2].plot(x, metrics["mrr"], c="tab:orange", label="test")

    ax[1, 0].plot(x, metrics["hits_1"], c="tab:orange")
    ax[1, 1].plot(x, metrics["hits_3"], c="tab:orange")
    ax[1, 2].plot(x, metrics["hits_10"], c="tab:orange")

    if baselines:
        ax[0, 2].plot(x, baselines["mrr"] * baselines_y, color="red", ls="--", lw=1.5, label="baseline")
        ax[1, 0].plot(x, baselines["hits_1"] * baselines_y, color="red", ls="--", lw=1.5, label="baseline")
        ax[1, 1].plot(x, baselines["hits_3"] * baselines_y, color="red", ls="--", lw=1.5, label="baseline")
        ax[1, 2].plot(x, baselines["hits_10"] * baselines_y, color="red", ls="--", lw=1.5, label="baseline")
        ax[0, 2].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()
        ax[1, 2].legend()

    ax[0, 0].semilogy()
    ax[0, 1].semilogy()
    ax[0, 0].legend()

    clear_output(True)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.)
    plt.show()
