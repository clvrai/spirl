import numpy as np


def euclid_distance(state1, state2):
    return np.float64(np.linalg.norm(state1 - state2))      # rewards need to be np arrays for replay buffer


def sparse_threshold(state1, state2, thresh):
    return np.float64(euclid_distance(state1, state2) < thresh)


