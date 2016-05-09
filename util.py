import itertools

import numpy as np
import scipy.stats
import pytest

from hmm import HMM, LearningHMM, DirMultMatrix

def check_marginals(samples, marginals):
    n_samples = len(samples)
    counts = sum(samples)
    ps = []
    for observed, expected in zip(counts, marginals):
        p = scipy.stats.binom_test(observed, n_samples, expected)
        print("Observed %d/%d, expected %.4f, p = %.4f"
              % (observed, n_samples, expected, p))
        ps.append(p)
    assert all(p > 0.01/len(ps) for p in ps)

def align_sequences(x, y):
    if max(x) < max(y):
        x, y = y, x
    perms = itertools.permutations(set(x), 1 + max(y))
    return max(np.count_nonzero(x == np.array(perm)[y]) for perm in perms)

def basic_hmm():
    t = [[0.7, 0.3, 0],
         [0.3, 0.7, 0],
         [0.5, 0.5, 0]]
    e = [[0.9, 0.1],
         [0.2, 0.8],
         [0, 0]]
    obs = [0, 0, 1, 0, 0]
    return HMM(t, e, obs, start_state=2)

def cyclic_hmm():
    t = [[0.01, 0.99, 0.00, 0.00],
         [0.00, 0.01, 0.99, 0.00],
         [0.00, 0.00, 0.01, 0.99],
         [0.99, 0.00, 0.00, 0.01]]
    e = [[0.0, 0.5, 0.5],
         [0.6667, 0.1667, 0.1667],
         [0.5, 0.0, 0.5],
         [0.3333, 0.3333, 0.3333]]
    obs = range(800)
    h = HMM(t, e, obs, start_state=0)
    h.states, h.obs = h.sample_forward(800)
    return h

def cyclic_hmm_dirichlet():
    h = cyclic_hmm()
    t_generator = DirMultMatrix(np.full_like(h.t, 0.1))
    e_generator = DirMultMatrix(np.full_like(h.e, 0.95))
    dh = LearningHMM(t_generator, e_generator, h)
    return dh
