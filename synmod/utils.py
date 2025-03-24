"""Common utility functions"""
import logging
import json

import numpy as np


def get_logger(name, filename, level=logging.INFO):
    """Return logger configure to write to filename"""
    formatting = "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
    logging.basicConfig(level=level, filename=filename, format=formatting)  # if not already configured
    logger = logging.getLogger(name)
    return logger


class JSONEncoderPlus(json.JSONEncoder):
    """JSON-serialize numpy objects"""
    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")

def strtointlist(val):
    try:
        val = [int(x) for x in val.split(",")]
    except:
        raise ValueError()
    return val

def argstr_to_list(value, name, args):
    try:
        vals = value.split(",")
        if len(vals) == args.num_features:
            dat = np.array([float(x) for x in vals])
        elif len(vals) == 1:
            dat = np.array([float(value) for x in range(args.num_features)])
        else:
            raise IndexError(
                f"Argument '{name}' is not valid. Number of probabilities is {len(vals)} and should be either 1 or {args.num_features}.")
    except Exception as ex:
        if isinstance(ex, IndexError):
            raise ex
        else:
            raise Exception(f"Argument '{name}' is not numeric or incorrect number of values.")

    return dat


def discretize_categoricals(ts_sample, feature, feature_id):
    f_vals = ts_sample[feature_id]
    for time_point, val in enumerate(f_vals):
        which_lower = np.argwhere(feature.threshold <= val).max()
        # which_higher = np.argwhere(self.threshold > val).min()
        return which_lower



def generate_obs_masks(ts_sample, feature, feature_id):
   pass