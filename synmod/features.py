"""Feature generation"""

from abc import ABC

import numpy as np

from synmod.constants import BINARY, CATEGORICAL, NUMERIC, CONSTANT
from synmod.generators import RandomWalk
from synmod.aggregators import Max, get_aggregation_fn_cls
from synmod.utils import argstr_to_list

class FeatureFeatureInteraction:
    def __init__(self, args, causal_feature, affected_feature):
        self.causal_feature_id = causal_feature.fid
        self.causal_feature: Feature = causal_feature
        self.affected_feature_id = affected_feature.fid
        self.affected_feature: Feature = affected_feature

        self.rng = affected_feature.generator._rng

        window_possible_start, window_possible_end = min(args.interact_range), max(args.interact_range)
        window_possible = list(range(window_possible_start, (window_possible_end + 1)))

        self.interaction_strength = self.rng.uniform(-args.interact_scaling, args.interact_scaling)

        window = self.rng.choice(window_possible, size=2)
        self.window_start = np.min(window)
        self.window_end = np.max(window)
        self.window_aggregation_function = ['mean', 'min', 'max'][self.rng.choice([0, 1, 2])]


class Feature(ABC):
    """Feature base class"""
    def __init__(self, name, seed_seq):
        self.name = name
        self._rng = np.random.default_rng(seed_seq)
        # Initialize relevance
        self.important = False
        self.effect_size = 0
        self.fid = -1

    def sample(self, *args, **kwargs):
        """Sample value for feature"""

    def summary(self):
        """Return dictionary summarizing feature"""
        return dict(name=self.name,
                    type=self.__class__.__name__)

class TemporalFeature(Feature):
    """Base class for features that take a sequence of values"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls):
        super().__init__(name, seed_seq)
        self.window = self.get_prediction_window(sequence_length)
        self.generator = None
        self.aggregation_fn = aggregation_fn_cls(**dict(rng=self._rng, window=self.window))
        # Initialize relevance
        self.window_important = False
        self.ordering_important = False
        self.window_ordering_important = False
        self.interactions = None

    def sample(self, *args, **kwargs):
        """Sample sequence from generator"""
        return self.generator.sample(*args, **kwargs)

    def summary(self):
        summary = super().summary()
        assert self.generator is not None
        summary.update(dict(window=self.window,
                            aggregation_fn=self.aggregation_fn.__class__.__name__,
                            generator=self.generator.summary()))
        return summary

    def predict(self, instances, window):
        preds = np.zeros_like(instances)
        for time in range(instances.shape[-1]):
            if time+window[1] >= 0:
                w_start = 0
                w_end = max(time + window[1], 0)
                if time+window[0] >= 0 and time+window[1] >= 0:
                    w_start = max(time+window[0], 0)
                preds[:,time] = self.aggregation_fn.operate(instances[:, w_start: w_end + 1]).flatten()
        return preds

    def sample_timepoint(self, args, time_point, ts_sample, feature_id, **kwargs):
        raw_value =  self.generator.sample(sequence_length=1)
        final_value = raw_value.item()

        if self.interactions is not None:
            for interaction in self.interactions:
                # We ignore values that are outside of the measured time series on account of burn-in period
                if time_point + interaction.window_start >= 0:

                    # inter_fid = interaction['inter_fid'].item()
                    inter_subsample = ts_sample[interaction.causal_feature_id, time_point + interaction.window_start:time_point + interaction.window_end + 1].flatten()

                    if interaction.window_aggregation_function == 'mean':
                        for i, x in enumerate(inter_subsample):
                            contribution = (x * interaction.interaction_strength) / len(inter_subsample)
                            final_value += contribution
                    elif interaction.window_aggregation_function == 'min':
                        min_loc = np.argmin(inter_subsample)
                        contribution = (inter_subsample[min_loc] * interaction.interaction_strength)
                        final_value += contribution
                    elif interaction.window_aggregation_function == 'max':
                        max_loc = np.argmax(inter_subsample)
                        contribution = (inter_subsample[max_loc] * interaction.interaction_strength)
                        final_value += contribution
                    else:
                        raise NotImplementedError()

        return final_value


    def get_prediction_window(self, sequence_length):
        """Randomly select a window for the feature where the model should operate in"""
        assert sequence_length is not None  # TODO: handle variable-length sequence case
        if sequence_length == 1:
            return (1, 1)  # tabular features
        # TODO: allow soft-edged windows (smooth decay of influence of feature values outside window)
        left = -self._rng.choice(range(1, int(sequence_length)//2))
        right = 0
        return (left, right)


class BinaryFeature(TemporalFeature):
    """Binary feature"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls, **kwargs):
        super().__init__(name, seed_seq, sequence_length, aggregation_fn_cls)
        generator_class = RandomWalk
        kwargs["n_categories"] = 2

        self.generator = generator_class(self._rng, BINARY, self.window, **kwargs)


class CategoricalFeature(TemporalFeature):
    """Categorical feature"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls, **kwargs):
        super().__init__(name, seed_seq, sequence_length, aggregation_fn_cls)
        generator_class = RandomWalk
        kwargs["n_categories"] = kwargs.get("n_states", self._rng.integers(3, 5, endpoint=True))
        self.generator = generator_class(self._rng, CATEGORICAL, self.window, **kwargs)


class ConstantFeature(TemporalFeature):
    """Constant feature"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls, **kwargs):
        super().__init__(name, seed_seq, sequence_length, aggregation_fn_cls)
        generator_class = RandomWalk
        self.generator = generator_class(self._rng, CONSTANT, self.window, **kwargs)
        self.constant_value = None

    def sample(self, *args, **kwargs):
        """Custom constant sampling - only sample once"""
        return self.generator.sample(*args, **kwargs)


class NumericFeature(TemporalFeature):
    """Numeric feature"""
    def __init__(self, name, seed_seq, sequence_length, aggregation_fn_cls, **kwargs):
        super().__init__(name, seed_seq, sequence_length, aggregation_fn_cls)
        generator_class = RandomWalk
        self.generator = generator_class(self._rng, NUMERIC, self.window, **kwargs)


def get_feature(args, fid):
    """Return randomly selected feature"""
    seed_seq = args.rng.bit_generator._seed_seq.spawn(1)[0]  # pylint: disable = protected-access
    name = str(fid)
    aggregation_fn_cls = get_aggregation_fn_cls(args.rng)

    kwargs = {"window_independent": args.window_independent}
    kwargs['trend_start_prob'] = args.trend_start_prob
    kwargs['trend_stop_prob'] = args.trend_stop_prob
    kwargs['trend_strength'] = args.trend_strength
    kwargs['variance_scaling'] = argstr_to_list(args.variance_scaling, 'variance_scaling', args)[fid]
    kwargs['observation_prob'] = argstr_to_list(args.observation_prob, 'observation_prob', args)[fid]

    feature_class = args.rng.choice([BinaryFeature, CategoricalFeature, NumericFeature, ConstantFeature], p=args.feature_type_distribution)
    if aggregation_fn_cls is Max:
        # Avoid low-variance features by sampling numeric or high-state-count categorical feature
        feature_class = args.rng.choice([CategoricalFeature, NumericFeature], p=[0.25, 0.75])
        # if feature_class == CategoricalFeature:
        #     kwargs["n_states"] = args.rng.integers(4, 5, endpoint=True)

    feature = feature_class(name, seed_seq, args.expected_seq_length, aggregation_fn_cls, **kwargs)
    args.logger.info(f"Generating feature class {feature_class.__name__} with window {feature.window} and"
                     f" aggregation_fn {aggregation_fn_cls.__name__}")

    return feature
