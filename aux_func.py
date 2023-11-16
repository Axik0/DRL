"""contains base agent class, annealing rules and env hacks,
this content has to be imported and used in a Jupyter notebook"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import imageio
import os
import time
import numpy as np
from IPython import display
from collections import deque

import torch
import torch.nn as nn
import torchsummary as summary
import tqdm.notebook as tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def rr(env, capture=False):
    """quick render fix assuming env's render_mode='rgb_array' """
    arr = env.render()
    if capture:
        return arr
    else:
        plt.axis('off')
        plt.imshow(arr)


def act(action, env):
    """filters unnecessary output"""
    return env.step(action)[:3]


class RandAgent:
    """baseline agent that performs random (sample from uniform distr) actions"""

    def __init__(self, env, aid_to_str=False, capture_path='./animations'):
        self.env = env
        try:
            self.n_states = self.env.observation_space.n
        except AttributeError:
            self.n_states = None
        try:
            self.n_actions = self.env.action_space.n
        except AttributeError:
            self.n_actions = None
        self.aid_to_str = aid_to_str  # action symbols (to be shown on policy graph)
        self.log = []  # container for anything related to learning process (to be shown on a graph)
        self.rendered_frames = None  # container for rendered images (numpy arrays) used by '.capture' method
        self.capture_path = capture_path  # storage folder for animation
        self.fly = None  # method to be executed throughout a trajectory (e.g. change policy based on local MDP context)
        if self.n_actions and self.n_states:
            self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions  # uniform policy initialisation (sa)

    def act(self, state):
        """draws a random sample from current distribution on actions"""
        action = np.random.choice(self.n_actions, p=self.policy[state])  # outputs integers thus doesn't require .item()
        return action

    def walk(self, max_length, render=False, **interkwargs):
        """perform max_length actions by agent N"""
        states, actions, rewards = [], [], []
        state = self.env.reset()[0]  # initial state
        q6 = deque(maxlen=6)  # prepare a fixed-size queue as 2-step MDP (sarsar) storage
        for i in range(max_length):
            # perform an action
            action = self.act(state)
            new_state, reward, done = act(action, env=self.env)
            # log
            states.append(state)  # append OLD state, everything breaks if you start from new
            actions.append(action)
            rewards.append(reward)
            # add (append) current step (sar) to queue
            q6.extend((state, action, reward))
            state = new_state
            if interkwargs and self.fly:
                try:
                    self.fly(q6, **interkwargs)
                except AttributeError:
                    print("intermediary update method hasn't been defined for this class")
            # continuous visualization w/ proper interrupt
            if render:
                try:
                    if render == 2:
                        arr = rr(capture=True, env=self.env)
                        self.rendered_frames.append(arr)
                    else:
                        rr(env=self.env)
                        display.display(plt.gcf())
                        time.sleep(1e-4)
                        display.clear_output(wait=True)
                except KeyboardInterrupt:
                    break
            if done:
                break
        trajectory = {'s': states,
                      'a': actions,
                      'r': rewards, }
        return trajectory

    # VISUALIZATION
    def capture(self, max_length, fps=30):
        """captures render output and creates gif animation from frames (rgb arrays)"""
        self.rendered_frames = []
        results = self.walk(max_length, render=2)
        name = self.env.unwrapped.spec.id + '.gif'
        path = os.path.join(self.capture_path, name)
        imageio.mimsave(path, ims=self.rendered_frames, fps=fps)
        print(f"Trajectory with reward {sum(results['r'])} has been captured to {path}")

    def learning_curve(self, title="", label="", l_scale=False, show_max=True, ax=None):
        """plots log vs iterations of algorithm"""
        ax = sns.lineplot(self.log, linewidth=1.0, ax=ax, label=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('iterations - 1')
        if l_scale:
            ax.set_yscale('symlog')
        if show_max:
            ax.axhline(y=max(self.log), color='r', linestyle='-', linewidth=0.5)
        return ax

    def show_policy(self, label=None):
        """plots current policy matrix as a heatmap"""
        plt.figure(figsize=(2, 5))
        concise_actions = {'xticklabels': self.aid_to_str} if self.aid_to_str else {}
        ax = sns.heatmap(self.policy, **concise_actions, cbar=False)
        ax.set_title(label if label else 'Current policy')
        return ax


# ANNEALING
def lin_ann_rate(i, n_total, start=1):
    return start * (1 - i / (n_total - 1))


def exp_ann_rate(i, start=1, thr=0, la=2/300, drop=0):
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
    result = start * (1 + (1/alpha - 1) ** ((i - mid)/mid) )**(-1)
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
    [sns.lineplot(sgm_ann_rate(i, n/2, alpha=j, drop=0), linewidth=1, ax=ax, label=f"α={j:.3f}") for j in np.arange(1e-5, 1e-1, 2e-2)]
    [sns.lineplot(exp_ann_rate(i, thr=0, la=j/n, drop=0), linewidth=1, ax=ax, label=f"λ={j/100}") for j in np.arange(2, 100, 20)]


# auxillary functions
class AnnealingRate:
    def __init__(self, start=0.5, n_iterations=100, drop=0):
        self.str_name = 'Constant rate'
        self.start = start
        self.drop = drop
        self.n_total = n_iterations

    def __call__(self, i):
        return [self.start] * self.n_total

    def plot(self, n_iterations=100, ax=None):
        if hasattr(self, "__call__") and callable(self.__call__):
            i = np.arange(n_iterations)
            ax = sns.lineplot(self.__call__(i), linewidth=1, ax=ax)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"{self.str_name}", fontsize=15)
            ax.set_xlabel('iterations - 1')
            return ax
        else:
            raise NotImplementedError


class LinearAR(AnnealingRate):
    """Linear decay from start to zero"""

    def __init__(self, n_iterations=100, start=1, drop=0, thr=0):
        super().__init__(n_iterations=n_iterations, start=start, drop=drop)
        self.str_name = 'Linear decay'

    def __call__(self, i):
        return self.start * (1 - i / (self.n_total - 1))


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


if __name__ == '__main__':
    # anneal_comparison(200)
    # b = AnnealingRate()
    # l = LinearAR()
    # s = SigmoidalAR(al=1e-2, n_iterations=100)
    # ex = ExponentialAR(la=1e-2, n_iterations=100)
    # ex.plot()
    sample_noise()
    plt.show()
