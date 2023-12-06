import torch
import numpy as np

from src.utils.storage import StateDict
from time import perf_counter


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def filter_predictions(predictions, targets, filter):
    # removing all scores of actual true triplets in predictions but one we interested in.
    # useful for computing filtered MRR and Hits@k
    interest_prediction_vals = predictions.gather(1, filter)
    predictions[targets == 1] = 0
    targets[targets == 1] = 0
    return predictions.scatter_(1, filter, interest_prediction_vals), \
        targets.scatter_(1, filter, torch.ones(interest_prediction_vals.shape, device=targets.device))


def draw_plots(state: StateDict, baselines=None):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    f, ax = plt.subplots(ncols=3, nrows=2, figsize=(24, 9))
    ax[0, 0].set(title="NLL loss", xlabel="Epoches", ylabel="Loss")
    ax[0, 1].set(title="Mean riemann grad norm on train", xlabel="Epoches", ylabel="Norm")
    ax[0, 2].set(title="MRR", xlabel="Epoches", ylabel="MRR")
    ax[1, 0].set(title="Hits@1", xlabel="Epoches", ylabel="Hits@1")
    ax[1, 1].set(title="Hits@3", xlabel="Epoches", ylabel="Hits@3")
    ax[1, 2].set(title="Hits@10", xlabel="Epoches", ylabel="Hits@10")

    last_epoch = state.last_epoch
    x = np.arange(1, last_epoch + 1)
    baselines_y = np.ones(len(x))
    ax[0, 0].plot(x, state.losses.train, c="tab:blue", label="train")
    ax[0, 0].plot(x, state.losses.val, c="tab:green", label="val")
    ax[0, 0].plot(x, state.losses.test, c="tab:orange", label="test")

    ax[0, 1].plot(x, state.losses.norms, c="tab:red")
    ax[0, 2].plot(x, state.metrics.mrr.test, c="tab:orange", label="test")
    ax[0, 2].plot(x, state.metrics.mrr.val, c="tab:green", label="val")

    ax[1, 0].plot(x, state.metrics.hits_1.test, c="tab:orange", label="test")
    ax[1, 0].plot(x, state.metrics.hits_1.val, c="tab:green", label="val")

    ax[1, 1].plot(x, state.metrics.hits_3.test, c="tab:orange", label="test")
    ax[1, 1].plot(x, state.metrics.hits_3.val, c="tab:green", label="val")

    ax[1, 2].plot(x, state.metrics.hits_10.test, c="tab:orange", label="test")
    ax[1, 2].plot(x, state.metrics.hits_10.val, c="tab:green", label="val")

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


@torch.no_grad()
def get_rank_approximation(model, new_rank):
    device = model.E.weight.data.device
    old_rank = model.core.shape

    Q_R, R_R = torch.qr(model.R.weight.data)
    new_R = torch.randn((model.R.weight.data.shape[0], new_rank[0]), device=device)
    new_R[:, :old_rank[0]] = Q_R
    new_R = torch.linalg.qr(new_R)[0]

    Q_E, R_E = torch.qr(model.E.weight.data)
    new_E = torch.randn((model.E.weight.data.shape[0], new_rank[1]), device=device)
    new_E[:, :old_rank[1]] = Q_E
    new_E = torch.linalg.qr(new_E)[0]

    old_core = torch.einsum("ijk,ia,jb,kc->abc", model.core.data, R_R, R_E, R_E)
    new_core = torch.zeros((new_rank[0], new_rank[1], new_rank[1]))
    new_core.uniform_(-1e-5, 1e-5)
    new_core[:old_rank[0], :old_rank[1], :old_rank[2]] = old_core

    model.E.weight.data = new_E
    model.R.weight.data = new_R
    model.core.data = new_core
    model.rank = new_rank
    return model


class Timer:
    def __enter__(self):
        self.start = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.time = perf_counter() - self.start
        