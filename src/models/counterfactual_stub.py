import numpy as np


def ips(rewards, propensities):
    rewards = np.array(rewards, dtype=float)
    propensities = np.array(propensities, dtype=float)

    if np.any(propensities <= 0):
        raise ValueError("All propensities must be > 0")

    return np.mean(rewards / propensities)


def snips(rewards, propensities):
    rewards = np.array(rewards, dtype=float)
    propensities = np.array(propensities, dtype=float)

    if np.any(propensities <= 0):
        raise ValueError("All propensities must be > 0")

    weights = 1.0 / propensities
    return np.sum(rewards * weights) / np.sum(weights)


if __name__ == "__main__":
    rewards = [1, 0, 1, 1, 0]
    propensities = [0.4, 0.5, 0.25, 0.6, 0.2]

    print("IPS:", ips(rewards, propensities))
    print("SNIPS:", snips(rewards, propensities))