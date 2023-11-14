import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import imageio
import os
import time
import numpy as np
from IPython import display
from collections import deque


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

    def __init__(self, env, aid_to_str, capture_path='./animations'):
        self.env = env
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.aid_to_str = aid_to_str  # action symbols (to be shown on policy graph)
        self.log = []  # container for anything related to learning process (to be shown on a graph)
        self.rendered_frames = None  # container for rendered images (numpy arrays) used by '.capture' method
        self.capture_path = capture_path  # storage folder for animation
        self.fly = None  # method to be executed throughout a trajectory (e.g. change policy based on local MDP context)
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
        # name = ENV.unwrapped.spec.id + '.gif'
        name = 'default' + '.gif'
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