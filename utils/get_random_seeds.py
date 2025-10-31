import numpy as np


def get_random_seeds(size, seed=42):
    """Get a list of random numbers without duplicates.

    References:
    1. https://stackoverflow.com/questions/9755538/how-do-i-create-a-list-of-random-numbers-without-duplicates
    2. https://stackoverflow.com/questions/47742622/np-random-permutation-with-seed
    """
    assert 0 < size <= 1000
    return np.random.default_rng(seed=seed).permutation(1000)[:size]


if __name__ == "__main__":
    for i in range(10):
        l = get_random_seeds(10000, seed=i)
        print(l)
