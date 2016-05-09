import pytest

import numpy as np
import scipy.stats

from hmm import HMM

@pytest.fixture
def basic_hmm():
    t = [[0.7, 0.3, 0],
         [0.3, 0.7, 0],
         [0.5, 0.5, 0]]
    e = [[0.9, 0.1],
         [0.2, 0.8],
         [0, 0]]
    h = HMM(t, e)
    obs = [0, 0, 1, 0, 0]
    return (h, obs)

def test_sample_states_exact(basic_hmm, n_samples=1000):
    (h, obs) = basic_hmm
    samples = [h.sample_states_exact(obs, start_state=2)
               for _ in range(n_samples)]
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
    (h, obs) = basic_hmm
    states = h.sample_states_exact(obs, start_state=2)
    samples = [np.array(h.sample_states_slice(obs, start_state=2,
                                              states=states, iters=1))
               for _ in range(n_samples)]
    assert all(s in [0, 1] for sample in samples for s in sample)
    check_marginals(samples, [0.1327, 0.1796, 0.6925, 0.1796, 0.1327])
