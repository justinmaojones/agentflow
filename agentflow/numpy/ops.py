import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)


def onehot(x, depth):
    shape = list(x.shape) + [depth]
    y = np.zeros(shape)
    y[np.arange(len(x)), x] = 1.0
    return y.astype("float32")


def binarize(x, b):
    y = []
    for i in range(b):
        y.append(x % 2)
        x = x // 2
    return np.concatenate(y, axis=-1)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def clip(x, clip_to_min=None, clip_to_max=None):
    if clip_to_min is not None:
        x = np.maximum(x, clip_to_min)
    if clip_to_max is not None:
        x = np.minimum(x, clip_to_max)
    return x


def eps_greedy_noise(action, num_actions, eps=0.05):
    assert eps >= 0 and eps <= 1, "eps must be between 0 and 1, {eps} is invalid"
    return np.where(
        np.random.rand(*action.shape) > eps,
        action,
        np.random.choice(num_actions, size=action.shape),
    )


def eps_greedy_noise_from_logits(action_logits, eps=0.05):
    if action_logits.ndim == 1:
        random_action = np.random.choice(action_logits.shape[-1])
        noise = np.random.randn()
    else:
        random_action = np.random.choice(
            action_logits.shape[-1], size=action_logits.shape[:-1]
        )
        noise = np.random.randn(*random_action.shape)
    best_action = action_logits.argmax(axis=-1)
    choose_random = noise < eps
    return choose_random * random_action + (1 - choose_random) * best_action


def gumbel_softmax_noise(action_logits, temperature=1.0, eps=1e-4):
    annealed_logits = action_logits / temperature
    u = np.random.rand(*annealed_logits.shape)
    g = -np.log(-np.log(u))
    return (g + annealed_logits).argmax(axis=-1)
