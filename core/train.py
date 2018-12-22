# Train script
import os
import importlib
import tensorflow as tf

def execute(args):
    import dataset.dataset as ds
    model_module = "architecture.version" + str(args['model']) + ".model"
    arch = importlib.import_module(model_module)
    project_path = os.getcwd()
    ds_obj = ds(args['dataset'], project_path)
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(args['param'], project_path)
    param_dict = hp_obj.get_params()
    img_batch, label_batch = ds_obj(param_dict)
    model = arch.create_model(param_dict)
    # Train the network and save the result
