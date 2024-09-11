import json
from re import I
import numpy as np

from .accuracy_tool import (
    gen_micro_macro_result,
)


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)


def cv_output_function(data, config, *args, **params):
    """the general cv output function

    """
    x, y = data.values()
    k1, k2 = data.keys()
    return json.dumps({
        "avg " + k1: "{:<5}".format(f"{np.mean(x):<2.2f}"),
        "std " + k1: "{:<5}".format(f"{np.var(x):<2.2f}"),
        "avg " + k2: "{:<5}".format(f"{np.mean(y):<2.2f}"),
        "std " + k2: "{:<5}".format(f"{np.std(y, ddof=1):<2.2f}"),
    }, sort_keys=False)
