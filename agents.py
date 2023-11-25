"""contains universal base agent class for gym environments, has to be imported and used in a Jupyter notebook"""

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

from aux_func import LinearAR, ExponentialAR, G_noise

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RandAgent:
    """baseline agent that performs random (sample from uniform distribution) actions in a given gym environment
        with discrete action and observation spaces by default (attributes are None if that's not applicable)"""
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
            self.policy = np.ones(
                (self.n_states, self.n_actions)) / self.n_actions  # uniform policy initialisation (sa)

    def act(self, state):
        """draws a random sample from current distribution on actions"""
        action = np.random.choice(self.n_actions, p=self.policy[state])  # outputs integers thus doesn't require .item()
        return action

    def render(self, capture=False):
        """plots environment at current state, assumes that its render_mode='rgb_array'"""
        arr = self.env.render()
        if capture:
            return arr
        else:
            plt.axis('off')
            plt.imshow(arr)

    def walk(self, max_length, render=False, **interkwargs):
        """perform max_length actions by agent N"""
        states, actions, rewards = [], [], []
        state = self.env.reset()[0]  # initial state
        q6 = deque(maxlen=6)  # prepare a fixed-size queue as 2-step MDP (sarsar) storage
        for i in range(max_length):
            # perform an action
            action = self.act(state)
            new_state, reward, done = self.env.step(action)[:3]
            # log
            states.append(state)  # append OLD state, everything breaks if you start from new
            actions.append(action)
            rewards.append(reward)
            # add (append) current step (sar) to queue
            q6.extend((state, action, reward))
            state = new_state
            if interkwargs and self.fly:
                try:
                    self.fly(q6, done, **interkwargs)
                except AttributeError:
                    print("intermediary update method hasn't been defined for this class")
            # continuous visualization w/ proper interrupt
            if render:
                try:
                    if render == 2:
                        arr = self.render(capture=True)
                        self.rendered_frames.append(arr)
                    else:
                        self.render()
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

    def __repr__(self):
        return (f"Agent {self.__class__.__name__} with {self.n_actions} actions "
                f"and {'discrete' if self.n_states else 'inf'} states initialised")


class ModelFreeAgent(RandAgent):
    """Model-free agent == R, TP functions of ENV aren't provided and aren't (explicitly?) approximated"""

    def __init__(self, env, aid_to_str):
        super().__init__(env=env, aid_to_str=aid_to_str)
        self.stc = 0  # counts improvement steps

    def gi(self, q, eps, zero_mask=None):
        """epsilon-greedy policy improvement based on values of q(s,a), at each state,
        policy ~ not only take an action that maximizes given action-value function q,
        but also allow "the rest" actions happen with eps probability (shared)

        unvisited states have q=0 but that doesn't matter anything,
        as rewards are < 0, such states would impact argmax in vain,
        zero_mask removes them from computation"""
        if zero_mask is not None:
            q[zero_mask] = np.full(shape=q.shape, fill_value=0)[zero_mask]
            # q = np.ma.array(q, mask=zero_mask)
        # epsilon-greedy: let eps be ~"the rest" actions probability, split it between the all actions
        dummy = (1 - eps + eps / self.n_actions) * np.eye(self.n_actions) + (eps / self.n_actions) * (
                    np.ones(self.n_actions) - np.eye(self.n_actions))
        best_actions = np.argmax(q, axis=0)  # 1D array of 'best' actions
        # one-hot encoding of best_actions array (used to choose rows from dummy)
        self.stc += 1
        return dummy[best_actions]

    def fit(self, n_trajectories, max_length, alpha_d=0.5, gamma=0.99, eps_d=None, verbose=False):
        dh = display.display(display_id=True)
        eps_rates, alpha_rates = [], []
        if callable(alpha_d):
            alpha_d.n_total = n_trajectories
        if eps_d is not None:
            eps_d.n_total = n_trajectories
        else:
            eps_d = LinearAR(n_iterations=n_trajectories, start=1)
        for i, t in enumerate(range(n_trajectories)):
            # decrease eps throughout the loop
            eps = eps_d(i)
            # decrease alpha ~ learning rate
            alpha = alpha_d(i) if callable(alpha_d) else alpha_d
            # policy evaluation and improvement throughout trajectory
            results = self.walk(max_length=max_length, alpha=alpha, eps=eps, gamma=gamma)
            # visualization
            eps_rates.append(eps)
            alpha_rates.append(alpha)
            self.log.append(sum(results['r']))
            if verbose:
                # ax = self.show_policy()
                ax = self.learning_curve(title=f"Total reward at trajectory of length < {max_length}", l_scale=False)
                ax2 = ax.twinx()
                ax2.tick_params(axis='y', labelcolor='blueviolet')
                ax2 = sns.lineplot(eps_rates, linewidth=0.5, ax=ax2, label="exploration, ε", color='blueviolet')
                sns.lineplot(alpha_rates, linewidth=0.5, ax=ax2, label="learning rate, α", color='orchid')
                dh.update(plt.gcf())
                plt.close()  # because plt.clf() is spurious


class CRandAgent(RandAgent):
    """baseline agent that performs random discrete actions (continuous environment)"""

    def __init__(self, env, aid_to_str=False):
        super().__init__(env=env, aid_to_str=aid_to_str)
        self.n_states = None
        self.d_states = self.env.observation_space.shape[0]
        self.policy = None

    def act(self, state):
        action = np.random.binomial(n=1, p=0.5, size=1)
        return action

    def show_policy(self, label=None):
        return None


class CrossEntropyNNCAgent(CRandAgent):
    """CrossEntropy algorithm actor, optimizes expected reward by policy, given by neural network"""
    def __init__(self, env, aid_to_str=False, noise_fn=G_noise, hidden_d=(120, None),
                 device=DEVICE):
        super().__init__(env=env, aid_to_str=aid_to_str)

        self.scale = 1
        self.noise_fn = noise_fn
        self.noise = torch.zeros(self.n_actions)

        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(in_features=self.d_states, out_features=hidden_d[0]),
            nn.ReLU(),
            nn.Linear(hidden_d[0], self.n_actions),
        )

    def act(self, state):
        """policy = action given by argmax (network output + scaled random noise)
            otherwise this policy would be almost completely deterministic (as NN is except its initialization)
            clipper is used instead of tanh due just to fit result into action space min max limits"""
        with torch.inference_mode():
            self.noise = self.noise_fn(dim=self.n_actions, pr3v=self.noise)
            # action = nn.functional.tanh(self.model(torch.Tensor(state)).detach() + self.scale * self.noise)
            action_d = self.model(torch.Tensor(state)).detach() + self.scale * self.noise
            action = torch.argmax(action_d)
        return action.numpy()

    def fit(self, n_iterations, max_n_epochs, noise_scale_d=None, lr=0.01, n_trajectories=100, max_length=50, q=0.8,
            verbose=None):
        """
        CE Algorithm has 2 steps per iteration: evaluate policy, improve policy
        Our goal is to maximize expected reward ER which is unreachable => approximated

        n_trajectories (w/ length <= max_length) defines quality of ER approximation
        0<q<1 controls rejected quantile of trajectories

        max_n_epochs controls max possible NN training epochs per each iteration

        lr defines learning rate of built-in Adam optimizer

        verbose>0 sets up a period of learning process rendering

        NB: .fit internally uses .act method of child class(this), doesn't inherit parental
        """
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        iterations_pbar = tqdm.trange(n_iterations, position=0, leave=True, colour="#a2d2ff")
        dh = display.display(display_id=True)
        if noise_scale_d is not None:
            noise_scale_d.n_total = n_iterations
        else:
            noise_scale_d = LinearAR(n_iterations=n_trajectories, start=self.scale)
        scales = []
        for i in iterations_pbar:
            self.model.eval()
            # policy evaluation (act with current policy or sample n_det deterministic from that one)
            trajectories = [self.walk(max_length=max_length) for t in range(n_trajectories)]
            rewards = np.stack([np.sum(t['r']) for t in trajectories])
            avg_reward = np.mean(rewards)
            self.log.append(avg_reward)
            # policy improvement
            # get q-quantile of current reward distribution and filter out better trajectories
            gamma = np.quantile(rewards, q)
            elite_ids = (rewards > gamma).nonzero()[0]
            if elite_ids.any():
                # extract (lists of) state and (corresponding) action tensors from elite trajectories
                states_l, actions_l = zip(
                    *((np.stack(trajectories[ei]['s']), np.stack(trajectories[ei]['a'])) for ei in elite_ids))
                states, actions = torch.Tensor(np.concatenate(states_l)).to(self.device), torch.tensor(
                    np.concatenate(actions_l)).to(self.device)
                epochs_pbar = [0] if (max_n_epochs == 1 or i == 0) else tqdm.trange(n_epochs, position=1, leave=False,
                                                                                    colour="#ffc8dd")  # order matters
                self.model.train()
                for e in epochs_pbar:
                    # forward pass
                    pred_actions_d = self.model(states)
                    loss = self.loss(pred_actions_d, actions)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # decrease noise scale
                self.scale = noise_scale_d(i)
                scales.append(self.scale)
                # less stochastic gradients (ascending from 1 to max_n_epochs)
                n_epochs_d = ExponentialAR(la=2 / n_iterations, start=1, drop=None)
                n_epochs = 1 + round((1 - n_epochs_d(i)) * (max_n_epochs - 1))
                iterations_pbar.set_postfix_str(
                    f'avg reward: {avg_reward.item():.2f}, loss: {loss.detach().item():.2e}, scaled_noise: {torch.mean(self.scale * self.noise).item():.2f}, n_epochs: {n_epochs}',
                    refresh=True)

            # visualization (plotting starts after at least 1 iteration)
            if verbose and i > 0 and (i + 1) % verbose == 0:
                # print(f"iteration {i + 1}, mean total reward: {avg_reward}")
                figure, axes = plt.subplots(1, 2, figsize=(12, 5))
                ax = self.learning_curve(ax=axes[0], title="Mean rewards")

                ax2 = ax.twinx()
                ax2.tick_params(axis='y', labelcolor='slateblue')
                sns.lineplot(scales, linewidth=0.5, ax=ax2, label="noise scale", color='slateblue')

                ax = sns.histplot(rewards, kde=False, bins=20, ax=axes[1])
                ax.axvline(gamma, 0, 20, color='r')
                ax.set_xlabel('Rewards')
                ax.set_title(f'Current distribution of rewards and its {q:.2f}-quantile ', fontsize=10)

                dh.update(plt.gcf())
                plt.close()  # because plt.clf() is spurious
        return avg_reward


class SARSAAgent(ModelFreeAgent):
    """SARSA updates at each step of every trajectory"""

    def __init__(self, env, aid_to_str):
        super().__init__(env=env, aid_to_str=aid_to_str)
        self.Q = np.zeros_like(self.policy.T)
        self.fly = self.sarsa_step

    def sarsa_step(self, queue_6, alpha, eps, gamma):
        """accepts a queue object, updates value-function q and policy in sarsa manner"""
        if len(queue_6) == 6:  # sarsar
            s, a, r, sx, ax = list(queue_6)[:5]
            self.Q[a, s] += alpha * (r + gamma * self.Q[ax, sx] - self.Q[a, s])
            self.policy = self.gi(self.Q, eps)


class QLearningAgent(ModelFreeAgent):
    """Q updates at each step of every trajectory"""

    def __init__(self, env, aid_to_str):
        super().__init__(env=env, aid_to_str=aid_to_str)
        self.Q = np.zeros_like(self.policy.T)
        self.fly = self.ql_step

    def ql_step(self, queue_6, alpha, eps, gamma):
        """accepts a queue object, updates value-function q and policy ~ Q-learning"""
        if len(queue_6) == 6:  # sarsar
            s, a, r, sx = list(queue_6)[:4]
            self.Q[a, s] += alpha * (r + gamma * np.max(self.Q[:, sx]) - self.Q[a, s])
            self.policy = self.gi(self.Q, eps)


class MonteCarloAgent(ModelFreeAgent):
    """Applies Monte-Carlo sampling of trajectories for action-value function q approximation"""

    def __init__(self, env, aid_to_str):
        super().__init__(env=env, aid_to_str=aid_to_str)

    def walk_r(self, gamma, max_length):
        """transforms standard results after tracing a route, yields
            n (matrix a,s of visited state counts),
            g (matrix a,s of returns)"""
        g, n = np.zeros_like(self.policy.T), np.zeros_like(self.policy.T)
        trajectory = super().walk(max_length=max_length)
        R, A, S = trajectory['r'], trajectory['a'], trajectory['s']
        # get rewards serie (summands, w/ discounting)
        gamma_ = np.cumprod(np.concatenate((np.atleast_1d(1.), np.tile(np.array(gamma), len(R) - 1))))
        R_discounted = gamma_ * np.array(R)
        # get values of returns G, e.g. total reward as if we'd started at timestep t and continued MDP on this trajectory
        G = np.array([np.sum(R_discounted[t:]) / gamma ** t for t in range(len(R_discounted))])
        # get unique pairs from 2D array, their places and counts
        unp, pos, cts = np.unique(np.vstack([A, S]), axis=1, return_inverse=True, return_counts=True)
        # row vector of stacked Gt values @ matrix with (stacked as rows ~ pos) one-hot encoded unique states => row vector with sum of Gts at unique states
        g[*unp], n[*unp] = G @ (np.eye(len(cts))[pos]), cts
        return g, n, sum(R)

    def fit(self, n_trajectories, max_length, eps_d=None, gamma=1, verbose=False):
        """MC approach replaces expected value calculation with an empirical mean of return Gt"""
        dh = display.display(display_id=True)
        if eps_d is not None:
            eps_d.n_total = n_trajectories
        else:
            eps_d = LinearAR(n_iterations=n_trajectories, start=1)
        eps_rates = []
        # np.full(shape=self.policy.T.shape, fill_value=-np.inf)
        G, N = np.zeros_like(self.policy.T), np.zeros_like(self.policy.T)
        for i, t in enumerate(range(n_trajectories)):
            # MC policy evaluation to approximate Q
            g, n, r = self.walk_r(max_length=max_length, gamma=gamma)
            G, N = G + g, N + n
            Q = G / (N_ := N.copy() + 1E-2 * (N == 0))  # prevent zero-division error

            # decrease eps throughout the loop
            eps = eps_d(i)

            # policy improvement (non-visited sa pairs are treated different)
            self.gpi(Q, eps=eps, zero_mask=N == 0)

            # visualization
            self.log.append(r)
            eps_rates.append(eps)
            if verbose:
                # ax = self.show_policy()
                ax = self.learning_curve(title=f"Total reward at trajectory of length < {max_length}", l_scale=False)
                ax2 = ax.twinx()
                ax2.tick_params(axis='y', labelcolor='blueviolet')
                sns.lineplot(eps_rates, linewidth=0.5, ax=ax2, label="exploration, ε", color='blueviolet')
                dh.update(plt.gcf())
                plt.close()  # because plt.clf() is spurious