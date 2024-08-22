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
    psnr, ssim = data.values()
    return json.dumps({
        "avg psnr": f"{sum(psnr) / len(psnr):<2.2f}",
        "min psnr": f"{min(psnr):<2.2f}",
        "avg ssim": f"{sum(ssim) / len(ssim) * 100:<2.2f}%",
        "min ssim": f"{min(ssim) * 100:<2.2f}%",
    }, sort_keys=True)
