import json
from re import I

from .accuracy_tool import (
    gen_micro_macro_result,
    calculate_ssim,
    calculate_psnr
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
        "average psnr": sum(psnr) / len(psnr),
        "average ssim": sum(ssim) / len(ssim)
    }, sort_keys=True)
