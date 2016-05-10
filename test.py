import numpy as np
import scipy.stats

from util import check_marginals, align_sequences
from util import basic_hmm, cyclic_hmm, cyclic_hmm_dirichlet, cyclic_hmm_hdp

def test_sample_states_exact(n_samples=1000):
    h = basic_hmm()
    samples = []
    for _ in range(n_samples):
        h.sample_states_exact()
        samples.append(np.array(h.states))
    assert all(s in [0, 1] for sample in samples for s in sample)
    check_marginals(samples, [0.1327, 0.1796, 0.6925, 0.1796, 0.1327])

def test_sample_states_slice(n_samples=1000):
    h = basic_hmm()
    h.sample_states_exact()
    samples = []
    for _ in range(n_samples):
        h.sample_states_slice()
        samples.append(np.array(h.states))
    assert all(s in [0, 1] for sample in samples for s in sample)
    check_marginals(samples, [0.1327, 0.1796, 0.6925, 0.1796, 0.1327])

def test_cyclic_hmm_fixed():
    h = cyclic_hmm()
    states = np.array(h.states)
    h.sample_states_exact()
    assert np.count_nonzero(states == h.states) > 750

def test_cyclic_hmm_dirichlet_stationary():
    dh = cyclic_hmm_dirichlet()
    h = dh.hmm
    states = np.array(h.states)
    dh.initialize_with_states(h.states)
    for _ in range(10):
        print(align_sequences(states, h.states))
        dh.sample_gibbs(1)
    assert align_sequences(states, h.states) > 750

def test_cyclic_hmm_dirichlet_convergence():
    dh = cyclic_hmm_dirichlet()
    h = dh.hmm
    states = np.array(h.states)
    dh.initialize_with_states(np.random.choice(range(4), size=states.size))
    for _ in range(10):
        print(align_sequences(states, h.states))
        dh.sample_gibbs(50)
    assert align_sequences(states, h.states) > 700

def test_cyclic_hmm_dir_slice_convergence():
    dh = cyclic_hmm_dirichlet()
    h = dh.hmm
    states = np.array(h.states)
    dh.initialize_with_states(np.random.choice(range(4), size=states.size))
    for _ in range(10):
        print(align_sequences(states, h.states))
        dh.sample_gibbs(50, sample_states_method=h.sample_states_slice)
    assert align_sequences(states, h.states) > 700

def test_cyclic_hmm_hdp_slice_stationary():
    hdh = cyclic_hmm_hdp()
    h = hdh.hmm
    states = np.array(h.states)
    hdh.initialize_with_states(h.states)
    for _ in range(100):
        print(align_sequences(states, h.states))
        hdh.sample_gibbs(1)

def test_cyclic_hmm_hdp_slice_convergence():
    hdh = cyclic_hmm_hdp()
    h = hdh.hmm
    states = np.array(h.states)
    hdh.initialize_with_states(np.random.choice(range(1, 21), size=states.size))
    for _ in range(10):
        print(align_sequences(states, h.states))
        hdh.sample_gibbs(50)
    assert align_sequences(states, h.states) > 700
