"""Feature temporal aggregation functions"""

from abc import ABC

import numpy as np
import torch


class Aggregator():
    """Aggregates temporal values"""
    def __init__(self, aggregation_fns, windows, instances=None, standardize_features=False):
        self._aggregation_fns = aggregation_fns  # Temporal aggregation functions for all features
        self._windows = windows  # Windows over time for all features (list of tuples)
        if instances is None:
            return
        # Identify statistics to standardize each feature
        num_features = len(self._aggregation_fns)
        self._means = np.zeros(num_features)
        self._stds = np.ones(num_features)
        if standardize_features:
            self.update_statistics(instances)

    def update_statistics(self, instances):
        """Identify statistics to standardize each feature"""
        for fidx, _ in enumerate(self._aggregation_fns):
            left, right = self._windows[fidx]
            vec = self.operate_on_feature(fidx, instances[:, fidx, left: right + 1])
            self._means[fidx] = np.mean(vec)
            self._stds[fidx] = np.std(vec)
            if self._stds[fidx] < 1e-10:
                # FIXME: features can pass the variance test earlier but fail it here, since the samples used are different
                self._stds[fidx] = 1

    def operate_on_feature(self, fidx, sequences, seq_start):
        """Operate on sequences for given feature"""
        # return (self._aggregation_fns[fidx].operate(sequences) - self._means[fidx]) / self._stds[fidx]  # sequences: instances X timesteps
        results, credit = self._aggregation_fns[fidx].operate(sequences, seq_start)
        return results, credit

    def operate(self, sequences):
        """Apply feature-wise operations to sequence data"""
        # TODO: when perturbing a feature, other values do not need to be recomputed.
        # But this seems unavoidable under the current design (analysis only calls model.predict, doesn't provide other info)
        num_instances, num_features, num_timepoints = sequences.shape  # sequences: instances X features X timesteps
        #matrix = np.zeros((num_instances, num_features))
        matrix = torch.zeros_like(sequences)
        credits = np.zeros_like(sequences.detach(), dtype=object)
        for fidx in range(num_features):
            (left, right) = self._windows[fidx]
            for time in range(num_timepoints):
                if time + right >= 0:
                    w_start = max(time + left, 0)
                    w_end = max(time + right, 0)
                    # if time + left >= 0 and time + right >= 0:
                    #     w_start =
                    res, credit = self.operate_on_feature(fidx, sequences[:, fidx, w_start: w_end + 1], seq_start=w_start)
                    matrix[:, fidx, time] = res
                    credits[:, fidx, time] = credit
        return matrix, credits


def apply_weights(seq, *weights):
    return seq.dot(weights[0])


class AggregationFunction(ABC):
    """Aggregation function base class"""
    #TODO: Restore the values
    # NONLINEARITY_OPERATORS = [lambda x: x, np.abs, np.square]
    NONLINEARITY_OPERATORS = [lambda x: x]

    def __init__(self, rng, window):
        self._nonlinearity_operator = rng.choice(AggregationFunction.NONLINEARITY_OPERATORS)
        self._window = window
        self._sequence_operator = None
        self.ordering_important = False
        self.weights = None

    # def operate(self, sequences):
    #     """Operate on sequences for given feature"""
    #     weights_to_use = self.weights if self.weights is None else self.weights[:sequences.shape[1]]
    #     return self._nonlinearity_operator(np.apply_along_axis(self._sequence_operator, 1, sequences, weights_to_use))  # sequences: instances X timesteps
    def operate(self, sequences, seq_start):
        """Operate on sequences for given feature"""
        weights_to_use = self.weights if self.weights is None else self.weights[:sequences.shape[1]]
        # val = self._nonlinearity_operator(torch.apply_along_axis(self._sequence_operator, 1, sequences, weights_to_use))  # sequences: instances X timesteps
        val = (sequences * torch.tensor(weights_to_use)).sum(axis=-1)
        credit_locs = np.stack([[seq_start + x for x in range(sequences.shape[-1])] for i in range(len(sequences))])
        credits = [(credit_locs[i], weights_to_use) for i in range(len(credit_locs))]

        return val, credits

class Max(AggregationFunction):
    """Computes max of inputs"""
    def __init__(self, rng, window):
        super().__init__(rng, window)
        self._sequence_operator = np.max
    def operate(self, sequences, seq_start):
        """Operate on sequences for given feature"""
        weights_to_use = self.weights if self.weights is None else self.weights[:sequences.shape[1]]
        val = self._nonlinearity_operator(np.apply_along_axis(self._sequence_operator, 1, sequences, weights_to_use))  # sequences: instances X timesteps

        credit_locs = seq_start + np.apply_along_axis(np.argmax, 1, sequences, weights_to_use) #since max, each credit loc gets 100% of the credit
        credits = [[(credit_locs[i], 1)] for i in range(len(credit_locs))]

        return val, credits

class Average(AggregationFunction):
    """Computes average of inputs"""
    def __init__(self, rng, window):
        super().__init__(rng, window)
        # self._sequence_operator = np.average
        window_size = window[1] - window[0] + 1
        self._sequence_operator = apply_weights
        self.weights = np.ones(window_size) / window_size

class MonotonicWeightedAverage(AggregationFunction):
    """Computes weighted average of inputs with monotically increasing weights"""
    def __init__(self, rng, window):
        super().__init__(rng, window)
        window_size = window[1] - window[0] + 1
        self.weights = np.linspace(1, 2, window_size)
        self._sequence_operator = apply_weights#lambda seq: seq.dot(self.weights)
        self.ordering_important = window_size > 1

class RandomWeightedAverage(AggregationFunction):
    """Computes weighted average of inputs with random weights"""
    def __init__(self, rng, window):
        super().__init__(rng, window)
        window_size = window[1] - window[0] + 1
        self.weights = np.linspace(1, 2, window_size)
        rng.shuffle(self.weights)
        self._sequence_operator = apply_weights#lambda seq: seq.dot(self.weights)
        self.ordering_important = window_size > 1


# AGGREGATION_OPERATORS = [Max, Average, MonotonicWeightedAverage, RandomWeightedAverage]
AGGREGATION_OPERATORS = [Average, MonotonicWeightedAverage, RandomWeightedAverage]


def get_aggregation_fn_cls(rng):
    """Sample aggregation function for feature"""
    return rng.choice(AGGREGATION_OPERATORS)
