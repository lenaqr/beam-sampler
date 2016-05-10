import numpy as np

def categorical(p):
    return p.cumsum().searchsorted(np.random.uniform(0, p.sum()))

def careful_dirichlet(alpha, size=None):
    try:
        ret = np.random.dirichlet(alpha, size=size)
        if np.any(np.isnan(ret)):
            raise ZeroDivisionError
        return ret
    except ZeroDivisionError:
        # apparently this happens sometimes with small alpha parameters
        # sigh
        if size is None:
            size = ()
        elif not isinstance(size, tuple):
            size = (size,)
        axis = len(size)
        u = np.random.random(size + np.shape(alpha))
        logx = np.log(u) / alpha
        logm = np.max(logx, axis=axis, keepdims=True)
        logx -= logm
        ret = np.exp(logx - np.log(np.sum(np.exp(logx), axis=axis, keepdims=True)))
        if np.any(np.isnan(ret)):
            import pdb; pdb.set_trace()
        return ret

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

    def set_params(self, t, e):
        self.t = np.asarray(t)
        self.e = np.asarray(e)
        (K, M) = self.e.shape
        assert (K, K) == self.t.shape
        self.K = K
        self.M = M

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

    def add_counts(self, t_generator, e_generator, incr=1):
        """Count the state transitions and emissions in the data."""

        t_generator.incorporate(self.states[:-1], self.states[1:], incr)
        t_generator.incorporate(self.start_state, self.states[0], incr)
        if self.end_state is not None:
            t_generator.incorporate(self.states[-1], self.end_state, incr)

        # TODO: handle missing observations
        e_generator.incorporate(self.states, self.obs, incr)

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
        self.hmm.add_counts(self.t_generator, self.e_generator)
        self.sample_params()

    def initialize_with_params(self, t, e):
        self.hmm.t = self.t_generator.params = t
        self.hmm.e = self.e_generator.params = e
        self.hmm.sample_states_exact()
        self.hmm.add_counts(self.t_generator, self.e_generator)

    def sample_gibbs(self, iters=1, sample_states_method=None):
        for _ in range(iters):
            self.sample_states(sample_states_method)
            self.sample_params()

    def sample_states(self, sample_states_method=None):
        self.hmm.add_counts(self.t_generator, self.e_generator, -1)
        if sample_states_method is not None:
            sample_states_method()
        else:
            self.hmm.sample_states_exact()
        self.hmm.add_counts(self.t_generator, self.e_generator)

    def sample_params(self):
        self.t_generator.sample_params()
        self.e_generator.sample_params()
        self.hmm.t = self.t_generator.params
        self.hmm.e = self.e_generator.params

class DirMultMatrix(object):
    def __init__(self, alpha, params, counts=None):
        """A matrix with a Dirichlet prior on each row.

        alpha: length-K vector of Dirichlet hyperparameters (pseudocounts)
        params: K-by-M matrix of sampled parameters
        counts: K-by-M matrix of observed counts

        """
        self.alpha = alpha
        self.params = params
        if counts is None:
            counts = np.zeros_like(params)
        self.counts = counts

    def sample_params(self):
        self.params = np.apply_along_axis(
            np.random.dirichlet, 1, self.alpha + self.counts)

    def incorporate(self, x, y, incr=1):
        np.add.at(self.counts, (x, y), incr)

    def reduce_rows(self, keep):
        """Discard rows corresponding to unrepresented states.

        keep: A length-K boolean array indicating the states to keep.

        """
        assert np.all(self.counts[~keep, :] == 0)
        self.counts = self.counts[keep, :]
        self.params = self.params[keep, :]

    def extend_rows(self, n_rows):
        """Add extra rows for new states."""

        (K, M) = self.params.shape
        n_new = n_rows - K
        if n_new <= 0:
            return

        rows_new = np.random.dirichlet(np.full(M, self.alpha), size=n_new)
        self.params = np.r_[self.params, rows_new]

        counts = np.zeros_like(self.params)
        counts[:K, :M] = self.counts
        self.counts = counts

class HDPMatrix(object):
    def __init__(self, gamma, alpha, beta, params, counts=None):
        """An infinite matrix with an HDP prior on each row.

        """
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.params = params
        if counts is None:
            counts = np.zeros_like(params)
        self.counts = counts

    def incorporate(self, x, y, incr=1):
        np.add.at(self.counts, (x, y), incr)

    def reduce_rows(self, keep):
        """Discard rows and cols corresponding to unrepresented states.

        keep: A length-K boolean array indicating the states to keep.

        """
        assert np.all(self.counts[~keep, :] == 0)
        assert np.all(self.counts[:, ~keep] == 0)
        self.counts = self.counts[keep, :][:, keep]
        self.params = self.params[keep, :][:, keep]
        self.beta = self.beta[keep]

    def sample_params(self):
        beta_full = np.r_[self.beta, 1 - np.sum(self.beta)]
        counts_full = np.c_[self.counts, np.zeros(self.counts.shape[0])]
        params_full = np.apply_along_axis(
            np.random.dirichlet, 1, self.alpha * beta_full + counts_full)
        self.params = params_full[:, :-1]

    def sample_beta(self):
        # sample auxiliary counts m_ij by simulating the CRP urn process
        (K, M) = self.counts.shape
        aux_counts = np.zeros(M)
        for j in range(M):
            weight = self.alpha * self.beta[j]
            probs = weight/(weight + np.arange(self.counts[:, j].max()))
            for i in range(K):
                count = int(self.counts[i, j])
                aux_counts[j] += np.count_nonzero(np.random.random(size=count) < probs[:count])
        # sample beta conditioned on m_ij
        gamma_full = np.r_[aux_counts, self.gamma]
        beta_full = np.random.dirichlet(gamma_full)
        self.beta = beta_full[:-1]

    def extend_rows(self, n_rows):
        """Add extra rows for new states."""

        (K, M) = self.params.shape
        assert K == M
        n_new = n_rows - K
        if n_new <= 0:
            return

        # break more sticks
        beta_rest = 1 - np.sum(self.beta)
        beta_new = []
        for _ in range(n_new):
            b = np.random.beta(1, self.gamma) * beta_rest
            beta_rest -= b
            beta_new.append(b)
        self.beta = np.r_[self.beta, beta_new]

        # sample new params conditioned on the new sticks
        params_rest = 1 - np.sum(self.params, axis=1)
        params_new = careful_dirichlet(self.alpha * np.r_[beta_new, beta_rest], size=K) * np.c_[params_rest]
        self.params = np.c_[self.params, params_new[:, :-1]]

        # sample new rows corresponding to the new sticks
        rows_new = np.random.dirichlet(self.alpha * np.r_[self.beta, beta_rest], size=n_new)
        self.params = np.r_[self.params, rows_new[:, :-1]]

        # what is going on?
        if np.any(np.isnan(self.params)):
            import pdb; pdb.set_trace()

        # extend the counts matrix with zeros
        counts = np.zeros_like(self.params)
        counts[:K, :M] = self.counts
        self.counts = counts

    def extend_slice(self, u_min):
        """Add extra states until the remaining probability is lower than
        u_min."""

        for i in range(1, 100):
            params_rest = 1 - np.sum(self.params, axis=1)
            if np.all(params_rest < u_min):
                break
            self.extend_rows(i + len(self.beta))

class HDPHMM(LearningHMM):
    def __init__(self, t_generator, e_generator, hmm):
        """An HMM with a prior over the transition and emission parameters,
        which can be learned from data.

        t_generator: the transition prior, an instance of HDPMatrix
        e_generator: the emission prior, an instance of HDPMatrix
        hmm: the initial HMM, an instance of HMM

        """
        self.t_generator = t_generator
        self.e_generator = e_generator
        self.hmm = hmm

    def initialize_with_states(self, states):
        self.hmm.states = states
        n_states = 1 + max(states)
        self.t_generator.extend_rows(n_states)
        self.e_generator.extend_rows(n_states)
        self.hmm.add_counts(self.t_generator, self.e_generator)
        self.sample_params()

    def sample_states(self, sample_states_method=None):
        assert sample_states_method is None or sample_states_method is self.hmm.sample_states_slice
        # decrement counts
        self.hmm.add_counts(self.t_generator, self.e_generator, -1)
        # sample u
        slices = self.hmm.sample_slices()
        u_min = min(slices)
        # sample additional pi, phi given u
        self.t_generator.extend_slice(u_min)
        self.e_generator.extend_rows(len(self.t_generator.params))
        self.hmm.set_params(self.t_generator.params, self.e_generator.params)
        # sample s given u
        self.hmm.sample_states_given_slices(slices)
        # increment counts
        self.hmm.add_counts(self.t_generator, self.e_generator)

    def sample_params(self):
        # relabel states
        K = len(self.t_generator.params)
        keep = np.zeros(K, dtype=bool)
        keep[self.hmm.states] = True
        keep[self.hmm.start_state] = True
        if self.hmm.end_state is not None:
            keep[self.hmm.end_state] = True
        self.t_generator.reduce_rows(keep)
        self.e_generator.reduce_rows(keep)
        state_map = np.full(K, np.count_nonzero(keep), dtype=int)
        state_map[np.where(keep)] = np.arange(np.count_nonzero(keep))
        self.hmm.states = state_map[self.hmm.states]
        # sample params
        self.t_generator.sample_params()
        self.e_generator.sample_params()
        self.hmm.t = self.t_generator.params
        self.hmm.e = self.e_generator.params
        # sample beta
        self.t_generator.sample_beta()
