"""Master pipeline"""


import argparse
import copy
import math
import functools
import json
import os
import pickle

import cloudpickle
import numpy as np

from synmod import constants
from synmod import features as F
from synmod import models as M
from synmod.features import ConstantFeature, CategoricalFeature, BinaryFeature
from synmod.utils import get_logger, JSONEncoderPlus, strtobool, strtointlist, discretize_categoricals, \
    generate_obs_masks


def synthesize(**kwargs):
    """API to synthesize features, data and model"""
    strargs = []
    for key, value in kwargs.items():
        strargs.append(f"-{key}")
        strargs.append(f"{value}")
    return main(strargs=strargs)


def main(strargs=None):
    """Parse args and launch pipeline"""
    parser = argparse.ArgumentParser("python synmod")
    # Required arguments
    required = parser.add_argument_group("Required parameters")
    required.add_argument("-output_dir", help="Output directory", required=True)
    required.add_argument("-num_features", help="Number of features",
                          type=int, required=True)
    required.add_argument("-num_instances", help="Number of instances",
                          type=int, required=True)
    required.add_argument("-synthesis_type", help="Type of data/model synthesis to perform",
                          choices=[constants.TEMPORAL], required=True)

    # Optional common arguments
    common = parser.add_argument_group("Common optional parameters")
    common.add_argument("-fraction_relevant_features", help="Fraction of features relevant to model",
                        type=float, default=1)
    common.add_argument("-num_model_interactions", help="number of pairwise in aggregation model (default 0)",
                        type=int, default=0)
    common.add_argument("-include_interaction_only_features", help="include interaction-only features in aggregation model"
                        " in addition to linear + interaction features (excluded by default)", type=strtobool)
    common.add_argument("-seed", help="Seed for RNG, random by default",
                        default=None, type=int)
    common.add_argument("-write_outputs", help="flag to enable writing outputs (alternative to using python API)",
                        type=strtobool)
    common.add_argument("-feature_type_distribution", help="option to specify distribution of binary/categorical/numeric"
                        "features types", nargs=4, type=float, default=[0.2, 0.2, 0.5, 0.1])

    # Temporal synthesis arguments
    temporal = parser.add_argument_group("Temporal synthesis parameters")

    # TODO: Make sequences dependent on windows by default to avoid unpredictability
    temporal.add_argument("-sequences_independent_of_windows", help="If enabled, Markov chain sequence data doesn't depend on timesteps being"
                          " inside vs. outside the window (default random)", type=strtobool, dest="window_independent")
    temporal.set_defaults(window_independent=None) #TODO: Fix
    temporal.add_argument("-model_type", help="type of model (classifier/regressor) - default random",
                          choices=[constants.CLASSIFIER, constants.REGRESSOR], default=constants.REGRESSOR)
    temporal.add_argument("-standardize_features", help="add feature standardization (0 mean, 1 SD) to model",
                          type=strtobool)

    #Observation parameters
    temporal.add_argument("-observation_prob", help="The probability of observing a given feature value at any time point. Can be "
                                                           "either a single probability applied to all features (i.e. '0.1') or a "
                                                           "comma-seperated list of probabilities (i.e. '0.1,0.4,0.9) corresponding "
                                                           "to each feature. Default is 1 applied to all  features.",
                          type=str, default="1.0") #TODO: Fix

    #Interaction parameters
    temporal.add_argument("-interactions_max", help="Maximum number of other features any one feature can interact with",
                          type=int, default=0) #TODO: Fix
    temporal.add_argument("-interactions_probability", help="The probability of one feature having an interaction with any other feature.",
                          type=float) #TODO: Fix
    temporal.add_argument("-interactions_range",
                          help="Defines the indices relative to the current time point from which the dependency window start point can be sampled. Defaults to '-5,1', ",
                          type=strtointlist, default='-5,-1') #TODO: Fix
    temporal.add_argument("-interactions_len",
                          help="The size of the interaction window (how many of the previous time points for feature A can influence the value of feature B)",
                          type=int) #TODO: Fix
    temporal.add_argument("-interaction_scale",
                          help="Specifies a maximum for the strength of dependencies.",
                          type=float,
                          default=0.1) #TODO: Fix

    temporal.add_argument("-expected_seq_length", help="Expected length of a sequence, sampled from a geometric distribution",
                          type=int) #TODO: Fix
    temporal.add_argument("-min_seq_len",
                          help="The minimum sequence length of any sampled sequence.",
                          type=int,
                          default=1) #TODO: Fix

    temporal.add_argument("-variance_scaling",
                          help="Scaler applied to the variance of the gaussian from which values are sampled. Either a float applied to all values, or a list of values (one per feature)",
                          type=str,
                          default="1") #TODO: Fix
    temporal.add_argument("-trend_start_prob",
                          help="Probability that an inactive trend will start at any given time point in a trend-enabled variable.",
                          type=float,
                          default=0.1) #TODO: Fix
    temporal.add_argument("-trend_stop_prob",
                          help="Probability that an active trend will stop at any given time point in a trend-enabled variable.",
                          type=float,
                          default=0.1) #TODO: Fix
    temporal.add_argument("-trend_strength",
                          help="Scaling factor for the .",
                          type=float,
                          default=0.1) #TODO: Fix

    args = parser.parse_args(args=strargs)
    if args.synthesis_type == constants.TEMPORAL:
        if args.expected_seq_length is None:
            parser.error(f"-expected_seq_length required for -synthesis_type {constants.TEMPORAL}")
        elif args.expected_seq_length <= 1:
            parser.error(f"-expected_seq_length must be greater than 1 for synthesis_type {constants.TEMPORAL}")
    else:
        args.expected_seq_length = 1
    return pipeline(args)


def configure(args):
    """Configure arguments before execution"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(args.seed)
    args.logger = get_logger(__name__, f"{args.output_dir}/synmod.log")
    if args.window_independent is None:
        args.window_independent = args.rng.choice([True, False])


def draw_visualize(features, instances, test_results, item_id=0, seq_lengths=None):
    import matplotlib.pyplot as plt
    import math
    n_plots_vert = 5 + 1
    n_plots_horiz = math.ceil(len(features) / n_plots_vert)
    figure, axis = plt.subplots(n_plots_vert, n_plots_horiz, figsize=(10, 10), layout='constrained')


    seq_len = instances.shape[-1]
    time_tick_lbls = [str(x) if x%5 == 0 else "" for x in range(seq_len)]

    for i in range(len(features)):
        f_name = features[i].name
        f_type = features[i].generator._feature_type
        top_descriptor = f"Feature {f_name} ({f_type})"
        if n_plots_horiz > 1:
            working_subplot = axis[i // n_plots_horiz, i % n_plots_horiz]
        else:
            working_subplot = axis[i // n_plots_horiz]
        working_subplot.set_title(f"{top_descriptor}")
        working_subplot.set_xticks(list(range(0, len(time_tick_lbls))), labels=time_tick_lbls)

        values = instances[item_id, i]
        working_subplot.plot(list(range(0, len(time_tick_lbls))), values, color='b',label='Instance')
        working_subplot.set_ylabel("Value", color='b')
        working_subplot.tick_params(axis='y', colors='b')

        # overlay_plot = working_subplot.twinx()
        # overlay_plot.bar(list(range(0, len(time_tick_lbls))), explan_res[:, i].flatten(), color='r', label='Predictions')
        # overlay_plot.set_ylabel("Predictions", color='r')
        # overlay_plot.tick_params(axis='y', colors='r')
        #
        # working_subplot.set_zorder(overlay_plot.get_zorder() + 1)
        # working_subplot.patch.set_visible(False)

    i = len(features)
    if n_plots_horiz > 1:
        working_subplot = axis[i // n_plots_horiz, i % n_plots_horiz]
    else:
        working_subplot = axis[i // n_plots_horiz]
    working_subplot.set_title(f"Predictions")
    actual_len = seq_lengths[item_id]
    actual_labels = time_tick_lbls[:actual_len]
    working_subplot.set_xticks(list(range(0, len(actual_labels))), labels=actual_labels)

    results = test_results[item_id]
    working_subplot.bar(list(range(0, len(actual_labels))), results[:actual_len], color='r', label='Predictions')
    working_subplot.set_ylabel("Predictions", color='r')
    working_subplot.tick_params(axis='y', colors='r')

    figure.savefig(f"Example_Instance{item_id}.png")
    plt.close()

def pipeline(args):
    """Pipeline"""
    configure(args)
    args.logger.info(f"Begin generating sequence data with args: {args}")
    features = generate_features(args)
    instances, masks = generate_instances(args, features)
    model = M.get_model(args, features, instances)
    gt_predictions, gt_contributors = ground_truth_estimation(args, features, instances, model)
    write_outputs(args, features, instances, model)

    model_instances = copy.deepcopy(instances)
    model_instances = np.nan_to_num(model_instances, 0)
    test_results = model.predict(model_instances)
    draw_visualize(features, instances, test_results, item_id=0, seq_lengths=seq_lengths)

    return features, instances, model, gt_predictions, gt_contributors


def assign_interfeature_dependencies(args, features):
    interaction_matrix = np.random.random([len(features), len(features)])
    np.fill_diagonal(interaction_matrix, 999, wrap=False) #make sure the diagonal is always greater than interaction probability so no feature can have an interaction with itself
    interaction_range = args.interactions_range
    window_possible_start, window_possible_end = min(interaction_range), max(interaction_range)
    window_possible = list(range(window_possible_start, (window_possible_end + 1)))

    for x in interaction_range:
        assert x<0, f"The argument 'interaction_range' with value {interaction_range} should consist of only 0 or negative numbers, i.e. -5,-1."

    for fid, feature in enumerate(features):
        f_rand_gen = feature.generator._rng
        interactions_locs = np.argwhere(interaction_matrix[fid] <= args.interactions_probability)
        if len(interactions_locs) > args.interactions_max:
            interactions_locs = f_rand_gen.choice(interactions_locs, args.interactions_max)

        interactions = []
        for int_loc in interactions_locs:
            # the important window should be relative to the end of sequence - i.e. relative to discharge or mortality for IHM prediction

            interaction_scale = f_rand_gen.uniform(-args.interaction_scale, args.interaction_scale)
            window_start = f_rand_gen.choice(window_possible, size=1)[0]
            window_end = min(-1, window_start + args.interactions_len)
            window_aggregation_fn_idx = f_rand_gen.choice([0,1,2])
            window_aggregation_function = ['mean', 'min', 'max'][window_aggregation_fn_idx]
            interactions.append({'inter_fid':int_loc, 'inter_scale':interaction_scale, 'w_start':window_start, 'w_end':window_end, 'w_fn':window_aggregation_function})

        feature.interactions = interactions


def generate_features(args):
    """Generate features"""
    features = [None] * args.num_features
    fid = 0
    while fid < args.num_features:
        #Generate a feature
        feature = F.get_feature(args, fid)

        features[fid] = feature
        fid += 1

    if args.interactions_max > 0:
        assign_interfeature_dependencies(args, features)

    return features


def sample_time_series(args, features, generation_length, seq_length, **kwargs):
    #for each time point, generate a value for each of the features
    ts_sample = np.zeros((len(features), generation_length))

    for time_point in range(generation_length):
        for feature_id, feature in enumerate(features):
            # mask = np.random.choice([np.nan, 1], size=cur_seq_len, p=[1 - features[feature_id].observation_probability, features[feature_id].observation_probability])
            val = feature.sample_timepoint(args, time_point, ts_sample, feature_id, **kwargs)
            ts_sample[feature_id, time_point] = val

    #TODO: discretize categoricals here
    # TODO: generate masks
    masks = []
    for feature_id, feature in enumerate(features):
        if isinstance(feature, ConstantFeature): #set the constant feature value to be whatever was first sampled
            ts_sample[feature_id, :] = ts_sample[feature_id, 0]
        elif isinstance(feature, CategoricalFeature) or isinstance(feature, BinaryFeature):
            ts_sample = discretize_categoricals(ts_sample, feature, feature_id)

        masks.append(generate_obs_masks(ts_sample, feature, feature_id, seq_length))

    return ts_sample, masks


def predict_time_series():
    pass


def generate_instances(args, features):
    """Generate instances"""
    #Sample the sequence lengths
    seq_lengths = np.zeros(args.num_instances)

    #Ensure that all sequences at least as long as the minimum length
    under_min_locs = np.argwhere(seq_lengths < args.min_seq_len).flatten()
    while len(under_min_locs) > 0:
        seq_lengths[under_min_locs] = np.random.geometric(p=(1/args.expected_seq_length), size=len(under_min_locs))
        under_min_locs = np.argwhere(seq_lengths < args.min_seq_len).flatten()
    # seq_lengths += args.min_seq_len-1

    max_len = np.max(seq_lengths).item()

    #Add time points to cover the born in period
    burn_in_time = -min(args.interactions_range)
    generate_length = max_len + burn_in_time

    instances = []
    masks = []
    for instance_id in range(args.num_instances):
        instance, mask = sample_time_series(args, features, generate_length, seq_length=seq_lengths[instance_id])
        instances.append(instance)
        masks.append(np.stack(mask))
    return np.stack(instances), np.stack(masks)


def generate_labels(model, instances):
    """Generate labels"""
    # TODO: decide how to handle multivariate case
    # TODO: joint generation of labels and features
    return model.predict(instances)


def ground_truth_estimation(args, features, instances, model):
    """Estimate and tag ground truth importance of features"""
    # pylint: disable = protected-access
    args.logger.info("Begin estimating ground truth effects")

    ground_truth_preds, _, realized_values, model_contribs = model.predict(instances)

    ground_truth_importance = np.zeros_like(model_contribs)

    for i_instance, instance in enumerate(data_credit):
        for i_time, time_values in enumerate(instance):
            for i_feat, data_contribs_to_feat_time in enumerate(time_values):
                model_val = model_contribs[i_instance, i_feat, i_time]
                for xx in range(len(data_contribs_to_feat_time)):
                    data_took_from_f = data_contribs_to_feat_time[xx]['fid']
                    data_took_from_l = data_contribs_to_feat_time[xx]['loc']
                    data_val_for_loc = data_contribs_to_feat_time[xx]['val']
                    ground_truth_importance[i_instance, data_took_from_f, data_took_from_l] += model_val * data_val_for_loc

    #normalize importances by instance using min-max scaling
    for loc, inst in enumerate(ground_truth_importance):
        new_inst = (inst - inst.min()) / (inst.max() - inst.min())
        ground_truth_importance[loc] = new_inst
    return ground_truth_preds, ground_truth_importance


def write_outputs(args, features, instances, model):
    """Write outputs to file"""
    if not args.write_outputs:
        return
    with open(f"{args.output_dir}/{constants.FEATURES_FILENAME}", "wb") as features_file:
        cloudpickle.dump(features, features_file, protocol=pickle.DEFAULT_PROTOCOL)
    np.save(f"{args.output_dir}/{constants.INSTANCES_FILENAME}", instances)
    with open(f"{args.output_dir}/{constants.MODEL_FILENAME}", "wb") as model_file:
        cloudpickle.dump(model, model_file, protocol=pickle.DEFAULT_PROTOCOL)
    write_summary(args, features, model)


def write_summary(args, features, model):
    """Write summary of data generated"""
    config = dict(synthesis_type=args.synthesis_type,
                  num_instances=args.num_instances,
                  num_features=args.num_features,
                  sequence_length=args.expected_seq_length,
                  model_type=model.__class__.__name__,
                  sequences_independent_of_windows=args.window_independent,
                  fraction_relevant_features=args.fraction_relevant_features,
                  num_model_interactions=args.num_model_interactions,
                  include_interaction_only_features=args.include_interaction_only_features,
                  seed=args.seed)
    # pylint: disable = protected-access
    features_summary = [feature.summary() for feature in features]
    model_summary = {}
    if args.synthesis_type == constants.TEMPORAL:
        model_summary["windows"] = [f"({window[0]}, {window[1]})" if window else None for window in model._aggregator._windows]
        model_summary["aggregation_fns"] = [agg_fn.__class__.__name__ for agg_fn in model._aggregator._aggregation_fns]
        model_summary["means"] = model._aggregator._means
        model_summary["stds"] = model._aggregator._stds
    model_summary["relevant_features"] = model.relevant_feature_names
    model_summary["polynomial"] = model.sym_polynomial_fn.__repr__()
    summary = dict(config=config, model=model_summary, features=features_summary)
    summary_filename = f"{args.output_dir}/{constants.SUMMARY_FILENAME}"
    args.logger.info(f"Writing summary to {summary_filename}")
    with open(summary_filename, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, cls=JSONEncoderPlus)
    return summary


if __name__ == "__main__":
    main()
