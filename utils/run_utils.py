import os
from model.model_factory import get_model_factory

def get_model(model_type, exp_file):
    exp_file = os.path.join("yaml", exp_file)
    model_factory = get_model_factory(model_type)
    model = model_factory.create_model(exp_file)
    return model
