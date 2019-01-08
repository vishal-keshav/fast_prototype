"""
Generate visualization does following:
1. Prints the model graph with Netron library
    -> from model
    -> from checkpoint (with backprop nodes)
2. Plots the training graphs
3. TODO: More visualization from tensorboard
"""

import netron
import tensorflow as tf
import os
import importlib

def get_model(model_version, param_version, project_path):
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(param_version, project_path)
    param_dict = hp_obj.get_params()
    model_module_path = "architecture.version" + str(model_version) + ".model"
    model_module = importlib.import_module(model_module_path)
    inputs = tf.placeholder(tf.float32, shape = (1, 224, 224, 3))
    model = model_module.create_model(inputs, param_dict)
    return model

def execute(args):
    project_path = os.getcwd()
    #model = get_model(args.model, args.param, project_path)
    #sess = tf.Session()
    #tf.train.write_graph(sess.graph_def, project_path + "/debug", 'model.pbtxt')
    #netron.start(project_path+"/debug/model.pbtxt")
    tflite_path = project_path + "/result/model_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    netron.start(tflite_path+"/model_tflite.tflite")
