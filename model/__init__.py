from .cv.net import ProposedModel

model_list = {
    'Ours': ProposedModel,
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
