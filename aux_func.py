import matplotlib.pyplot as plt


def rr(env):
    """quick render fix assuming env's render_mode='rgb_array' """
    # plt.figure(figsize=(5,10))
    plt.axis('off')
    plt.imshow(env.render())


def act(action_id: int, env):
    """filters unnecessary output"""
    return env.step(action_id)[:3]


