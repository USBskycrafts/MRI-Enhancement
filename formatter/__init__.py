import logging

from .Basic import BasicFormatter
from .cv.NibabelFormmater import NibabelFormmater
from .nifti.NIFTI1Formmater import NIFTI1Formatter

logger = logging.getLogger(__name__)

formatter_list = {
    "Basic": BasicFormatter,
    "Nibabel": NibabelFormmater,
    "NIFTI": NIFTI1Formatter,
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error(
            "There is no formatter called %s, check your config." % which)
        raise NotImplementedError
