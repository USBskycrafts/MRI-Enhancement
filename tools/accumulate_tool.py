from typing import *
from .accuracy_init import init_accuracy_functions, calculate_mean_loss


def accumulate_cv_data(result: dict, acc_result: Union[dict, None], config, mode, *args, **params):
    functions = init_accuracy_functions(config, *args, **params)
    if acc_result is None:
        acc_result = {key: [] for key in functions.keys()}
    for key, values in acc_result.items():
        func = functions.get(key, None)
        if func is None:
            raise Exception(
                f"Error: {key} is not defined in accuracy functions or output")
        output, gt = result["output"], result["gt"]
        values.append(func(output, gt, config))
    return acc_result


def accumulate_loss(losses: Dict[str, float], acc_result: Union[Dict[str, List[float]], None], config, mode, *args, **params):
    if acc_result is None:
        acc_result = {key: [] for key in losses.keys()}
    for key, value in acc_result.items():
        value.append(losses[key])
    return acc_result
