"""Contains noise functions, annealing rates and useful functions,
this content has to be imported and used in a Jupyter notebook"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np

import torch
import torch.nn as nn
import torchsummary as summary
import tqdm.notebook as tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ANNEALING
def lin_ann_rate(i, n_total, start=1):
    return start * (1 - i / (n_total - 1))


def exp_ann_rate(i, start=1, thr=0, la=2 / 300, drop=0):
    """exponential decay with i --> +inf, provides descending values within range [expl_thr, start],
        la=0.01 means ~36% left after 100 iterations, results < drop value are zeroed"""
    result = thr + (start - thr) * np.exp(- la * i)
    if drop:
        if isinstance(i, np.ndarray):
            result[result <= drop] = 0
        elif result < drop:
            result = 0
    return result


def sgm_ann_rate(i, mid, start=1, alpha=1e-2, drop=0):
    """symmetric sigmoidal decay within [0, 2*mid-1] provides descending values within range [1-alpha, alpha],
    results < drop value are zeroed, alpha controls the gap between function value and its asymptotes (i=1/i=0)"""
    # establish max possible smoothness given alpha and symmetrize wrt midpoint (this form has been pre-simplified)
    result = start * (1 + (1 / alpha - 1) ** ((i - mid) / mid)) ** (-1)
    if drop:
        if isinstance(i, np.ndarray):
            result[result <= drop] = 0
        elif result < drop:
            result = 0
    return result


def anneal_comparison(n=500):
    i = np.arange(n)
    ax = sns.lineplot(lin_ann_rate(i, n), linewidth=1.2, label=f"base")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Annealing rates comparison", fontsize=15)
    ax.set_xlabel('iterations - 1')
    [sns.lineplot(sgm_ann_rate(i, n / 2, alpha=j, drop=0), linewidth=1, ax=ax, label=f"α={j:.3f}") for j in
     np.arange(1e-5, 1e-1, 2e-2)]
    [sns.lineplot(exp_ann_rate(i, thr=0, la=j / n, drop=0), linewidth=1, ax=ax, label=f"λ={j / 100}") for j in
     np.arange(2, 100, 20)]


# auxillary functions
class AnnealingRate:
    def __init__(self, start=0.5, n_iterations=100, drop=0):
        self.str_name = 'Constant rate'
        self.start = start
        self.drop = drop
        self.n_total = n_iterations

    def __call__(self, i):
        return ([self.start] * self.n_total)[i]

    def plot(self, n_iterations=100, ax=None):
        if hasattr(self, "__call__") and callable(self.__call__):
            idx = np.arange(n_iterations)
            graph = [self.__call__(i) for i in idx]
            ax = sns.lineplot(graph, linewidth=1, ax=ax)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"{self.str_name}", fontsize=15)
            ax.set_xlabel('iterations - 1')
            return ax
        else:
            raise NotImplementedError


class LinearAR(AnnealingRate):
    """Linear decay from start to zero"""

    def __init__(self, n_iterations=100, start=1, drop=0):
        super().__init__(n_iterations=n_iterations, start=start, drop=drop)
        self.str_name = 'Linear decay'

    def __call__(self, i):
        return self.start * (1 - i / (self.n_total - 1))


class SteppingAR(AnnealingRate):
    """Stepping decay from start to zero, given by percent values of value and size"""
    def __init__(self, n_iterations=100, start=1, drop=0, steps=((100, 20), (50, 40), (25, 20), (0, 20))):
        super().__init__(n_iterations=n_iterations, start=start, drop=drop)
        self.str_name = 'Stepping decay'
        assert np.array(steps)[:, 1].sum() == 100, f'wrong steps {steps}'
        self.stepping = []
        for (pv, s) in steps:
            self.stepping += [start*pv/100]*(int(n_iterations*s/100))
        self.stepping = np.array(self.stepping)

    def __call__(self, i):
        return self.stepping[i]


class ExponentialAR(AnnealingRate):
    """exponential decay with i --> +inf, provides descending values within range [expl_thr, start],
        la=0.01 means ~36% left after 100 iterations, results < drop value are zeroed"""

    def __init__(self, la, n_iterations=100, start=1, drop=1e-7, thr=0):
        super().__init__(n_iterations=n_iterations, start=start, drop=drop)
        self.str_name = 'Exponential decay'
        self.thr = thr
        self.la = la

    def __call__(self, i):
        result = self.thr + (self.start - self.thr) * np.exp(- self.la * i)
        if self.drop:
            if isinstance(i, np.ndarray):
                result[result <= self.drop] = 0
            elif result < self.drop:
                result = 0
        return result


class ExponentialRiseAR(ExponentialAR):
    """slow exponential rise with i --> +inf, provides ascending values within range [expl_thr, start],
        la=0.01 means ~36% left after 100 iterations, results < drop value are zeroed"""

    def __init__(self, la, n_iterations=100, end=1, drop=1e-7, thr=0):
        super().__init__(n_iterations=n_iterations, la=la, start=end, drop=drop, thr=thr)
        self.str_name = 'Slow exponential decay'

    def __call__(self, i):
        return 1 - super().__call__(i)


class SigmoidalAR(AnnealingRate):
    """symmetric sigmoidal decay within [0, 2*mid-1] provides descending values within range [1-alpha, alpha],
        results < drop value are zeroed, alpha controls the gap between function value and its asymptotes (i=1/i=0)"""

    def __init__(self, al, n_iterations=100, start=1, drop=1e-7, thr=0):
        super().__init__(n_iterations=n_iterations, start=start, drop=drop)
        self.str_name = 'Sigmoidal decay'
        self.thr = thr
        self.al = al

    def __call__(self, i):
        # establish max possible smoothness given alpha and symmetrize wrt mid value (this form has been pre-simplified)
        mid = self.n_total // 2
        result = self.start * (1 + (1 / self.al - 1) ** ((i - mid) / mid)) ** (-1)
        if self.drop:
            if isinstance(i, np.ndarray):
                result[result <= self.drop] = 0
            elif result < self.drop:
                result = 0
        return result


def U_noise(dim, pr3v=None):
    """multidimensional Uniform noise within interval -1, +1"""
    return torch.Tensor((dim,)).uniform_(-1, 1)


def G_noise(dim, pr3v=None):
    """multidimensional Gauss noise with mean=0 and std=1"""
    return torch.randn(dim)


def OU_noise(dim, pr3v):
    """multidimensional random noise generated by Ornstein–Uhlenbeck process"""
    dp = {'theta': 0.15, 'delta': 0.01, 'mu': torch.zeros(dim), 'sigma': 0.1 * torch.ones(dim),
          'eps': G_noise(dim)}  # default params for RL application
    n3xt = pr3v + (dp['mu'] - pr3v) * dp['theta'] * dp['delta'] + dp['sigma'] * torch.sqrt(torch.tensor(dp['delta'])) * \
           dp['eps']
    return n3xt


def sample_noise(dim=1, n_iterations=100, scale=(1, 20), show_limits=True, anneal=True):
    """plots consequent noise sample of size 100"""
    pr3v = torch.zeros(dim, )
    g, ou = [], []
    for i in range(n_iterations):
        if anneal:
            rate = sgm_ann_rate(i, mid=n_iterations / 2 - 1, alpha=1e-01)
        g.append(scale[0] * rate * G_noise(dim, pr3v))
        pr3v = OU_noise(dim, pr3v)
        ou.append(scale[1] * rate * pr3v)
    g, ou = np.array(g), np.array(ou)
    plt.plot(g, linewidth=1.0, label=f"Gauss noise, s={scale[0]}")
    plt.plot(ou, linewidth=1.2, label=f"OU noise, s={scale[1]}")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"", fontsize=10)
    ax.set_xlabel('iterations - 1')
    # ax.set_yscale('symlog')
    plt.legend()
    if show_limits:
        ax.axhline(y=max(g), color='r', linestyle='-', linewidth=0.5)
        ax.axhline(y=max(ou), color='g', linestyle='-', linewidth=0.5)
        ax.axhline(y=min(g), color='r', linestyle='-', linewidth=0.5)
        ax.axhline(y=min(ou), color='g', linestyle='-', linewidth=0.5)


def varlinspace(odd_func, **lspkwargs):
    """linspace with variable step size, takes same keyword arguments as np.linspace
        function must be odd and monotonic, e.g. power function lambda x: x**(3)"""
    eqd = np.linspace(**lspkwargs)
    res_ = odd_func(eqd)
    if np.isnan(res_).any():
        print("wrong function choice, results contain nans")
    elif np.isinf(res_).any():
        print("wrong function choice, results contain infinities")
    else:
        med = np.median(res_, axis=0)
        left, right = res_*(res_ < med[None, :]), res_*(res_ >= med[None, :])
        # scale left and right parts
        left_s, right_s = left*np.abs(np.min(eqd,axis=0)/np.min(left, axis=0)), right*np.abs(np.max(eqd, axis=0)/np.max(right, axis=0))
        # scale could have been different and lead to non-monotonic, have to sort it again
        res = np.sort(left_s + right_s, axis=0)
        # sns.scatterplot(res)
        sns.stripplot(res, jitter=1e-3, orient='h', s=6, marker="v")
        return res


if __name__ == '__main__':
    # anneal_comparison(200)
    # b = AnnealingRate()
    # l = LinearAR()
    # s = SigmoidalAR(al=1e-2, n_iterations=100)
    # ex = ExponentialAR(la=1e-2, n_iterations=100)
    # ex.plot()
    # sample_noise()
    # st = ExponentialRiseAR(la=0.3)
    # st.plot()
    ul = np.array([4.8, 4, 0.41887903, 5])
    dis = varlinspace(lambda x: x ** (3) + 0.2 * x, start=-ul / 2, stop=ul / 2, num=5)
    cfg = [dis[:, col] for col in range(dis.shape[1])]
    plt.show()

