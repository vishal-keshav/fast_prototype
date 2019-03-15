"""
This module is used for assessing the distribution hidden
in weights and activation of a neural network model.

We need four things in order:
1. The dataset with which we want to evaluate the activation
2. The model, that will have a dictionary with layer outputs
3. The testing or evaluation loop for data collection for every layer
4. Plotter for for each dataset
"""

import tensorflow as tf
import numpy as np

import os
import importlib


class channel_wise:
    def __init__(self):
        # We hardcode for now
        #data_provider_module_path = "dataset." + dataset + ".data_provider"
        data_provider_module_path = "dataset.IMAGENET.data_provider"
        data_provider_module = importlib.import_module(data_provider_module_path)
        dp_obj = data_provider_module.get_obj(project_path)
        dp_obj.set_train_batch(1)
        iter = dp_obj.get_input_output()
        self.imgs = iter[0]
        self.labels = iter[1]
        # We get the model
        #model_module_path = "architecture.version" + str(version) + ".model"
        model_module_path = "architecture.version3.model"
        model_module = importlib.import_module(model_module_path)
        model = model_module.create_model(inputs, param_dict)
        # Get plotter
        plotter_module_path = "debug.statistics"
        plotter_module = importlib.import_module(model_module_path)
        plotters = [model_module.distribution_plotter()]*model.nr_layers()

    def train_and_collect():
        pass

    def plot_distribution():
        pass
