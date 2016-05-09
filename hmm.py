import logging

import numpy as np
import scipy.special
import scipy.stats

logger = logging.getLogger(__name__)

# hmm: (t, e)
# state sequence: [ss]
# output sequence: [ys]
# sample states (uncollapsed): (t, e), [ys] -> [ss]
# transition counts: [ss] -> counter(s)
# observation counts: [ss], [ys] -> counter(o)
# sample params: alpha, counter(x) -> theta

# sample u: t, [ss] -> [us]
# sample extended elements of t and e
# sample s given u: (t, e), [us], [ys] -> [ss]
# sample phi, emission params: alpha_e, x -> theta
# sample pi, transition params: alpha_t, beta, counter(s) -> t
# sample m, aux counts: alpha_t, beta, counter(s) -> counter(o)
# sample beta, base params: gamma, counter(o) -> beta

def categorical(p):
    return p.cumsum().searchsorted(np.random.uniform(0, p.sum()))

class HMM(object):
    def __init__(self, t, e, i=None):
        self.t = np.asarray(t)
        self.e = np.asarray(e)
        self.i = np.asarray(i)

    def sample_states_exact(self, obs, start_state, end_state=None, out=None):
        """Sample a state sequence conditioned on a sequence of observations.

        obs: a sequence of N observations, n = [1..N] (may contain Nones for missing data)
        start_state: the initial state, n = 0
        end_state: the ending state, n = N+1 (may be None to not condition on the ending state)
        out: a length-N vector to store the sampled states n = [1..N]

        """
        N = len(obs)
        K, _ = self.e.shape

        # forward filtering
        factors = np.empty((N, K))
        factor = (start_state == np.arange(K))
        for n in range(N):
            factor = np.dot(factor, self.t)
            if obs[n] is not None:
                factor *= self.e[:, obs[n]]
            factor /= np.sum(factor)
            factors[n] = factor

        # backward sampling
        if out is None:
            out = np.empty(N, dtype=int)
        state = end_state
        for n in reversed(range(N)):
            factor = factors[n]
            if state is not None:
                factor = factor * self.t[:, state]
            state = categorical(factor)
            out[n] = state

        return out

    def sample_states_slice(self, obs, start_state, states, end_state=None, iters=1):
        for _ in range(iters):
            slices = self.sample_slices(states, start_state)
            self.sample_states_given_slices(obs, slices, start_state, end_state=end_state, out=states)
        return states

    def sample_slices(self, states, start_state):
        """Sample the slice variables needed for sample_states_slice.

        states: the sequence of N states, n = [1..N] on which depend
        the conditional distribution of the slice variables
        start_state: the initial state, n = 0

        """
        prev_states = np.concatenate(([start_state], states[:-1]))
        return np.random.uniform(0, self.t[prev_states, states])

    def sample_states_given_slices(self, obs, slices, start_state, end_state=None, out=None):
        """Sample a state sequence conditioned on a sequence of observations, using slice sampling to limit the pool of states.

        obs: a sequence of N observations, n = [1..N] (may contain Nones for missing data)
        slices: a sequence of N u-values, the auxiliary slice variables used to threshold the set of considered states
        start_state: the initial state, n = 0
        end_state: the ending state, n = N+1 (may be None to not condition on the ending state)
        out: a length-N vector to store the sampled states n = [1..N]

        """
        N = len(obs)
        K, _ = self.e.shape

        # forward filtering
        factors = np.empty((N, K))
        factor = (start_state == np.arange(K))
        for n in range(N):
            factor = np.dot(factor, slices[n] < self.t)
            if obs[n] is not None:
                factor = factor * self.e[:, obs[n]]
            factor /= np.sum(factor)
            factors[n] = factor

        # backward sampling
        if out is None:
            out = np.empty(N)
        state = end_state
        for n in reversed(range(N)):
            factor = factors[n]
            if state is not end_state:
                factor = factor * (slices[n+1] < self.t[:, state])
            elif state is not None:
                factor = factor * self.t[:, state]
            state = categorical(factor)
            out[n] = state

        return out

import pytest

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
