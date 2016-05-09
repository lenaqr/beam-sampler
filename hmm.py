import numpy as np

def categorical(p):
    return p.cumsum().searchsorted(np.random.uniform(0, p.sum()))

class HMM(object):
    def __init__(self, t, e, obs, start_state, end_state=None, states=None):
        """An HMM that explains a sequence of observations.

        t: K-by-K transition matrix
        e: K-by-M emission matrix
        obs: a sequence of N observations, n = [1..N] (may contain Nones for missing data)
        start_state: the initial state, n = 0
        end_state: the ending state, n = N+1 (may be None to not condition on the ending state)
        out: a length-N vector to store the sampled states n = [1..N]

        """
        self.t = np.asarray(t)
        self.e = np.asarray(e)
        self.obs = obs
        self.start_state = start_state
        self.end_state = end_state
        N = len(obs)
        (K, M) = self.e.shape
        assert (K, K) == self.t.shape
        self.N = N
        self.K = K
        self.M = M
        if states is None:
            states = np.empty(N, dtype=int)
        self.states = states

    def sample_states_exact(self):
        """Sample a state sequence using exact forward-backward sampling."""

        N = self.N
        K = self.K

        # forward filtering
        factors = np.empty((N, K))
        factor = (self.start_state == np.arange(K))
        for n in range(N):
            factor = np.dot(factor, self.t)
            if self.obs[n] is not None:
                factor *= self.e[:, self.obs[n]]
            factor /= np.sum(factor)
            factors[n] = factor

        # backward sampling
        state = self.end_state
        for n in reversed(range(N)):
            factor = factors[n]
            if state is not None:
                factor = factor * self.t[:, state]
            state = categorical(factor)
            self.states[n] = state

    def sample_states_slice(self, iters=1):
        """Sample a state sequence using slice (beam) sampling."""

        for _ in range(iters):
            slices = self.sample_slices()
            self.sample_states_given_slices(slices)

    def sample_slices(self):
        """Sample the slice variables needed for sample_states_slice."""

        prev_states = np.concatenate(([self.start_state], self.states[:-1]))
        return np.random.uniform(0, self.t[prev_states, self.states])

    def sample_states_given_slices(self, slices):
        """Sample a state sequence conditioned on a sequence of slice variables to limit the pool of states."""

        N = self.N
        K = self.K

        # forward filtering
        factors = np.empty((N, K))
        factor = (self.start_state == np.arange(K))
        for n in range(N):
            factor = np.dot(factor, slices[n] < self.t)
            if self.obs[n] is not None:
                factor = factor * self.e[:, self.obs[n]]
            factor /= np.sum(factor)
            factors[n] = factor

        # backward sampling
        state = self.end_state
        for n in reversed(range(N)):
            factor = factors[n]
            if state is not self.end_state:
                factor = factor * (slices[n+1] < self.t[:, state])
            elif state is not None:
                factor = factor * self.t[:, state]
            state = categorical(factor)
            self.states[n] = state
