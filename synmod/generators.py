"""Generator base class"""

from abc import ABC
from collections import namedtuple

import numpy as np
import graphviz
from scipy.stats import bernoulli

from synmod.constants import NUMERIC, CONSTANT

IN_WINDOW = "in-window"
OUT_WINDOW = "out-window"
SummaryStats = namedtuple("SummaryStats", ["mean", "sd"])


class Generator(ABC):
    """Sequence generator base class"""
    def __init__(self, rng, feature_type, window):
        self._rng = rng
        self._feature_type = feature_type
        self._window = window

    def sample(self, sequence_length):
        """Sample sequence of given length from generator"""

    def graph(self):
        """Graph representation of generator (dot file)"""

    def summary(self):
        """Summary of generator"""



# pylint: disable = invalid-name, pointless-statement
class RandomWalk(Generator):
    """Random Walk generator"""

    class State():
        """Random Walk state"""
        # pylint: disable = protected-access, too-many-instance-attributes
        def __init__(self, chain, index, state_type, **kwargs):
            self._chain = chain  # Parent Random Walk chain
            self._index = index  # state identifier
            self._state_type = state_type  # state type - in-window vs. out-window
            self.name = f"{self._state_type}-{self._index}"
            self._p = None  # Transition probabilities from state
            self._states = None  # States to transition to
            self.mean = None #Mean value of normal distribution from which values are sampled
            self.sd = None #Std Dev of normal distribution from which values are sampled
            self.sample = None  # Function to sample from state distribution
            if self._chain._feature_type == NUMERIC:
                self._summary_stats = SummaryStats(None, None)
            self.variance_scaler = kwargs.get("variance_scaler", 1)
            self.threshold = None
            self.bin_order = None

        def continuous_sample_fn(self):
            return lambda: self._chain._rng.normal(self.active_mean, self.active_sd)

        def discrete_sample_fn(self):
            def actualized_discrete_sampler():
                val = self._chain._rng.normal(self.active_mean, self.active_sd)
                which_lower = np.argwhere(self.threshold <= val).max()
                # which_higher = np.argwhere(self.threshold > val).min()
                return which_lower

            return lambda: actualized_discrete_sampler()

        def gen_distributions(self):
            """Generate state transition and sampling distributions"""
            feature_type = self._chain._feature_type
            rng = self._chain._rng
            self._state_object = self._chain._in_window_state if self._state_type == IN_WINDOW else self._chain._out_window_state

            mean = rng.uniform(-1, 1)
            sd = (rng.uniform(0.1) * 0.05) * self.variance_scaler
            self.base_mean = mean
            self.active_mean = mean
            self.base_sd = sd
            self.active_sd = sd

            self._summary_stats = SummaryStats(mean, sd)

            self.sample = self.continuous_sample_fn()
            # if feature_type == NUMERIC or feature_type == CONSTANT:
            #     self.sample = self.continuous_sample_fn()
            # else:  # binary/categorical variable
            #     self.sample = self.discrete_sample_fn()

        def transition(self):
            """Transition to next state"""
            assert self._state_object is not None, "gen_distributions must be invoked first"
            return self._state_object

    def __init__(self, rng, feature_type, window, **kwargs):
        super().__init__(rng, feature_type, window)

        self.n_categories = kwargs.get("n_categories", self._rng.integers(3, 5, endpoint=True))
        self._window_independent = kwargs.get("window_independent", False)  # Sampled state independent of window location

        # If trends enabled, sampled values increase/decrease/stay constant according to trends corresponding to each state:
        self._has_trends = self._rng.choice([True, False]) #if self._feature_type == NUMERIC else False
        self._init_value = self._rng.uniform(-1, 1)  # Initial value of Random Walk, used for trends
        self._trend_start_prob = np.random.uniform(0,kwargs['trend_start_scaler'])
        self._trend_stop_prob = np.random.uniform(0,kwargs['trend_stop_scaler'])
        self._trend_strength = np.random.uniform(0, 0.2)
        self._is_trending = False
        self._is_markov = False #TODO: Make this an option

        # Select states inside and outside window
        self._in_window_state = self.State(self, 0, IN_WINDOW, **kwargs)
        self._out_window_state = self._in_window_state
        self.n_states = 1

        if not self._window_independent:
            # Create separate chain in/out of window
            self._out_window_state = self.State(self, 0, OUT_WINDOW, **kwargs)
            self.n_states = 2

        states = [self._in_window_state] if self._window_independent else [self._in_window_state, self._out_window_state]
        for state in states:
            state.gen_distributions()

            if feature_type == 'binary' or feature_type == 'categorical':
                starting_category = state._chain._rng.choice(list(range(self.n_categories)))

                base_threshold = [(3* x * state.base_sd) for x in range(-starting_category, self.n_categories-starting_category-1)]
                state.threshold = [state.base_mean + (y ) for y in base_threshold]
                state.threshold = np.array([-np.inf] + state.threshold + [np.inf])

    def sample(self, sequence_length, **kwargs):
        cur_state = self._out_window_state  # initial state
        # value = self._init_value  # TODO: what if value is re-initialized for every sequence sampled? (trends)
        left, right = self._window

        sequence = np.empty(sequence_length)
        for timestep in range(sequence_length):
            if not self._window_independent:
                # Reset initial state in/out of window
                if timestep == left:
                    cur_state = self._in_window_state
                elif timestep == right + 1:
                    cur_state = self._out_window_state
            # Get value from state
            value = cur_state.sample()
            sequence[timestep] = value
            # Set next state
            cur_state = cur_state.transition()

            if self._is_trending:
                self._is_trending = self._rng.choice([True, False], p=[1-self._trend_stop_prob, self._trend_stop_prob])
            else:
                self._is_trending = self._rng.choice([True, False], p=[self._trend_start_prob, 1 - self._trend_start_prob])

            if self._is_trending:
                cur_state.active_mean = cur_state.active_mean + (self._trend_strength * value)
                # cur_state.active_mean = cur_state.active_mean + (self._trend_strength * (value - cur_state.base_mean))
            if not self._is_markov:
                cur_state.active_mean = value



        return sequence

    def graph(self):
        graph = graphviz.Digraph()
        label = f"Random Walk \nFeature type: {self._feature_type}"
        if self._has_trends:
            label += f"\nTrends: True\nInitial value: {self._init_value:1.5f}"
        left, right = self._window
        label += f"\nWindow: [{left}, {right}]\n\n"
        graph.attr(label=label, labelloc="t")
        clusters = [self._in_window_state]
        clabels = [""]
        if not self._window_independent:
            clusters.append(self._out_window_state)
            clabels = ["In-window states", "Out-of-window states"]
        for cidx, cluster in enumerate(clusters):
            with graph.subgraph(name=f"cluster_{cidx}") as cgraph:
                cgraph.attr(label=clabels[cidx])
                for state in cluster:
                    # pylint: disable = protected-access
                    label = f"State {state._index}"
                    if self._feature_type == NUMERIC:
                        label += f"\nMean: {state._summary_stats.mean:1.5f}\nSD: {state._summary_stats.sd:1.5f}"
                    cgraph.node(state.name, label=label)
                    for oidx, ostate in enumerate(cluster):
                        cgraph.edge(state.name, ostate.name, label=f" {state._p[oidx]:1.5f}\t\n")
        return graph

    def summary(self):
        summary = {}
        if self._feature_type == NUMERIC:
            summary["trends"] = self._has_trends
            if self._has_trends:
                summary["init_value"] = self._init_value
        for stype, states in {"out_window_states": self._out_window_state, "in_window_states": self._in_window_state}.items():
            states_summary = [None] * len(states)
            for idx, state in enumerate(states):
                state_summary = {}
                # pylint: disable = protected-access
                state_summary["index"] = state._index
                state_summary["p"] = state._p
                if self._feature_type == NUMERIC:
                    mean, sd = state._summary_stats
                    state_summary["stats"] = dict(mean=mean, sd=sd)
                states_summary[idx] = state_summary
            summary[stype] = states_summary
        return summary
