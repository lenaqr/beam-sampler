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

    def sample_forward(self, length):
        """Sample a random observation sequence from the model."""

        states = []
        obs = []
        state = self.start_state
        for _ in range(length):
            state = categorical(self.t[state, :])
            states.append(state)
            obs.append(categorical(self.e[state, :]))
        return states, obs

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

    def add_counts(self, t_counts=None, e_counts=None, incr=1):
        """Count the state transitions and emissions in the data."""

        if t_counts is None:
            t_counts = np.zeros_like(self.t)
        if e_counts is None:
            e_counts = np.zeros_like(self.e)

        np.add.at(t_counts, (self.states[:-1], self.states[1:]), incr)
        np.add.at(t_counts, (self.start_state, self.states[0]), incr)
        if self.end_state is not None:
            np.add.at(t_counts, (self.states[-1], self.end_state), incr)

        # TODO: handle missing observations
        np.add.at(e_counts, (self.states, self.obs), incr)

        return t_counts, e_counts

class LearningHMM(object):
    def __init__(self, t_generator, e_generator, hmm):
        """An HMM with a prior over the transition and emission parameters,
        which can be learned from data.

        t_generator: the transition prior, an instance of DirMultMatrix
        e_generator: the emission prior, an instance of DirMultMatrix
        hmm: the initial HMM, an instance of HMM

        """
        self.t_generator = t_generator
        self.e_generator = e_generator
        self.hmm = hmm

    def initialize_with_states(self, states):
        self.hmm.states = states
        self.hmm.add_counts(self.t_generator.counts, self.e_generator.counts)
        self.sample_params()

    def initialize_with_params(self, t, e):
        self.hmm.t = self.t_generator.params = t
        self.hmm.e = self.e_generator.params = e
        self.hmm.sample_states_exact()
        self.hmm.add_counts(self.t_generator.counts, self.e_generator.counts)

    def sample_gibbs(self, iters=1, sample_states_method=None):
        for _ in range(iters):
            self.sample_states(sample_states_method)
            self.sample_params()

    def sample_states(self, sample_states_method=None):
        t_counts = self.t_generator.counts
        e_counts = self.e_generator.counts
        self.hmm.add_counts(t_counts, e_counts, -1)
        if sample_states_method is not None:
            sample_states_method()
        else:
            self.hmm.sample_states_exact()
        self.hmm.add_counts(t_counts, e_counts)

    def sample_states_slice(self):
        t_counts = self.t_generator.counts
        e_counts = self.e_generator.counts
        self.hmm.add_counts(t_counts, e_counts, -1)
        self.hmm.sample_states_slice()
        self.hmm.add_counts(t_counts, e_counts)

    def sample_params(self):
        self.t_generator.sample_params()
        self.e_generator.sample_params()
        self.hmm.t = self.t_generator.params
        self.hmm.e = self.e_generator.params

class DirMultMatrix(object):
    def __init__(self, alpha, counts=None, params=None):
        """A matrix with a Dirichlet prior on each row.

        alpha: K-by-M matrix of Dirichlet hyperparameters (pseudocounts)
        counts: K-by-M matrix of observed counts
        params: K-by-M matrix of sampled parameters

        """
        self.alpha = alpha
        if counts is None:
            counts = np.zeros_like(alpha)
        self.counts = counts
        if params is None:
            params = np.empty_like(alpha)
        self.params = params

    def sample_params(self):
        self.params = np.apply_along_axis(
            np.random.dirichlet, 1, self.alpha + self.counts)

