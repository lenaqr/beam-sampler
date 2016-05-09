import pytest

import numpy as np
import scipy.stats

from hmm import HMM, LearningHMM, DirMultMatrix

@pytest.fixture
def basic_hmm():
    t = [[0.7, 0.3, 0],
         [0.3, 0.7, 0],
         [0.5, 0.5, 0]]
    e = [[0.9, 0.1],
         [0.2, 0.8],
         [0, 0]]
    obs = [0, 0, 1, 0, 0]
    return HMM(t, e, obs, start_state=2)

def test_sample_states_exact(basic_hmm, n_samples=1000):
    h = basic_hmm
    samples = []
    for _ in range(n_samples):
        h.sample_states_exact()
        samples.append(np.array(h.states))
    assert all(s in [0, 1] for sample in samples for s in sample)
    check_marginals(samples, [0.1327, 0.1796, 0.6925, 0.1796, 0.1327])

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

def test_sample_states_slice(basic_hmm, n_samples=1000):
    h = basic_hmm
    h.sample_states_exact()
    samples = []
    for _ in range(n_samples):
        h.sample_states_slice()
        samples.append(np.array(h.states))
    assert all(s in [0, 1] for sample in samples for s in sample)
    check_marginals(samples, [0.1327, 0.1796, 0.6925, 0.1796, 0.1327])

@pytest.fixture
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

def test_cyclic_hmm_fixed(cyclic_hmm):
    h = cyclic_hmm
    states = np.array(h.states)
    h.sample_states_exact()
    assert np.count_nonzero(states == h.states) > 750

import itertools

def align_sequences(x, y):
    if max(x) < max(y):
        x, y = y, x
    perms = itertools.permutations(set(x), 1 + max(y))
    return max(np.count_nonzero(x == np.array(perm)[y]) for perm in perms)

def test_cyclic_hmm_dir(cyclic_hmm):
    h = cyclic_hmm
    states = np.array(h.states)
    t_generator = DirMultMatrix(np.full_like(h.t, 0.1))
    e_generator = DirMultMatrix(np.full_like(h.e, 0.95))
    lh = LearningHMM(t_generator, e_generator, h)
    lh.initialize_with_states(np.random.choice(range(4), size=800))
    for _ in range(10):
        print(align_sequences(states, h.states))
        lh.sample_gibbs(50)
    assert align_sequences(states, h.states) > 700
