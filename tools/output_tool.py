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

    Args:
        data (dict): 指标函数所需要的输入
    """
    raise NotImplementedError("deprecated")
    output, target = data.values()
    psnr = calculate_psnr(output, target, config, psnr_result)
    ssim = calculate_ssim(output, target, config, ssim_result)
    return json.dumps({"psnr": psnr.item(), "ssim": ssim.item()}, sort_keys=True)


class VisionOutputTool:
    def __init__(self):
        self.history_result = {
            "psnr": [],
            "ssim": []
        }

    def cv_output_function(self, data, config, *args, **params):
        output, target = data.values()
        psnr = calculate_psnr(output, target, config,
                              self.history_result["psnr"])
        ssim = calculate_ssim(output, target, config,
                              self.history_result["ssim"])
        return json.dumps({"psnr": psnr.item(),
                           "ssim": ssim.item(),
                           "lowest pnsr": min(self.history_result["psnr"]),
                           "lowest ssim": min(self.history_result["ssim"])},
                          sort_keys=True)
