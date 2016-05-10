import itertools

import numpy as np
import scipy.stats
import pytest

from hmm import HMM, LearningHMM, HDPHMM, DirMultMatrix, HDPMatrix

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
    t_generator = DirMultMatrix(0.1, h.t)
    e_generator = DirMultMatrix(0.95, h.e) # NB: this is not actually specified in the paper
    dh = LearningHMM(t_generator, e_generator, h)
    return dh

def cyclic_hmm_hdp():
    h = cyclic_hmm()
    t_generator = HDPMatrix(3.8, 0.4, np.zeros(1), np.zeros((1, 1)))
    e_generator = DirMultMatrix(0.95, np.zeros((1, 3)))
    hdh = HDPHMM(t_generator, e_generator, h)
    return hdh
