import numpy as np

def categorical(p):
    return p.cumsum().searchsorted(np.random.uniform(0, p.sum()))

class HMM(object):
    def __init__(self, t, e):
        self.t = np.asarray(t)
        self.e = np.asarray(e)

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
