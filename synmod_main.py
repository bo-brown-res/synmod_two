def main():
    from synmod import master
    import argparse

    args = argparse.Namespace()
    args.num_instances = 10
    args.num_features = 10
    # args.sequence_length = 100
    args.model_type = "regressor"  # regressor/classifier
    args.fraction_relevant_features = 0.5
    args.window_independent = False  # Sequence dependence on windows
    args.output_dir = "synmod_outputs"
    args.seed = 42+6
    args.num_interactions = 0 #TODO: remove mentions
    args.include_interaction_only_features = False #TODO remove mentions
    args.synthesis_type = "temporal"  # Don't alter #TODO: remove

    args.trend_start_prob = [0.2] * args.num_features
    args.trend_stop_prob = [0.2] * args.num_features
    args.trend_strength = [0.2] * args.num_features
    args.variance_scaling = [0.2] * args.num_features
    args.observation_prob = [0.2] * args.num_features
    args.feature_type_distribution = [0.2, 0.2, 0.5, 0.1]
    args.num_model_interactions = [0.2, 0.2, 0.5, 0.1]
    args.standardize_features = False
    args.write_outputs = True

    args.expected_seq_length = 100
    args.min_seq_len = 10
    args.interactions_max = 5
    args.interact_range = [-5,-1]
    args.interact_scaling = 0.1
    args.interact_prob = 0.2


    features, data, model, true_labels, true_contrib_feat_idxs = master.pipeline(args)

if __name__ == "__main__":
    main()