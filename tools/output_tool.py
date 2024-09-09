import json
from re import I

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
        "avg " + k1: "{:<5}".format(f"{sum(x) / len(x):<2.2f}"),
        "min " + k1: "{:<5}".format(f"{min(x):<2.2f}"),
        "avg " + k2: "{:<5}".format(f"{sum(y) / len(y):<2.2f}"),
        "min " + k2: "{:<5}".format(f"{min(y):<2.2f}"),
    }, sort_keys=True)
