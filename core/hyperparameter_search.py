"""
Hyper-parameter search based on basian optimizers (Gaussin process)
scikit learn library skopt is implements the optimizer.
We build on top of it, the easy to execute hyper-parameter setting
before we star the full-blown training.
"""

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os
import sys
import importlib

def get_hyper_parameters(version, project_path, get_param_space = True):
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(version, project_path)
    param_dict = hp_obj.get_params()
    if get_param_space:
        param_dict_space = hp_obj.get_param_space()
        return param_dict, param_dict_space
    else:
        return param_dict

def set_hyper_parameter(version, project_path, dict):
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(version, project_path)
    for key, value in dict.items():
        hp_obj.set_parameter(key, value)
    hp_obj.dump_parameter()

default_param_name = []
default_param_value = []
space_dimensions = []
args = None

def sanity_check(param_dict, param_dict_space):
    global default_param_name
    global default_param_value
    global space_dimensions
    for key, value in param_dict.items():
        if key in param_dict_space.keys():
            default_param_name.append(key)
            default_param_value.append(value)
            param_space = param_dict_space[key]
            if param_space["type"] == "Int":
                temp_skopt_var = Integer(low=param_space["low"], high=param_space["high"], name=str(key))
            elif param_space["type"] == "Real":
                temp_skopt_var = Real(low=param_space["low"], high=param_space["high"], prior=param_space["prior"], name=str(key))
            else:
                temp_skopt_var = Categorical(categories=param_space["catagories"], name=str(key))
            space_dimensions.append(temp_skopt_var)
        else:
            print(key + " has no space defined in param_space")
    return

#TODO create a dataprovider that works on a sample of data
def get_data_provider(dataset, project_path, param_dict):
    data_provider_module_path = "dataset." + dataset + ".data_provider"
    data_provider_module = importlib.import_module(data_provider_module_path)
    dp_obj = data_provider_module.get_obj(project_path)
    with tf.name_scope('input'):
        img_batch = tf.placeholder(tf.float32, [None, 28, 28, 1])
    with tf.name_scope('output'):
        label_batch = tf.placeholder(tf.int64, [None, 10])
    return img_batch, label_batch, dp_obj

def get_model(version, inputs, param_dict):
    model_module_path = "architecture.version" + str(version) + ".model"
    model_module = importlib.import_module(model_module_path)
    model = model_module.create_model(inputs, param_dict)
    return model

def optimisation(label_batch, logits, param_dict):
    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=label_batch, logits=logits))
    tf.summary.scalar('loss', loss_op)
    with tf.name_scope('gradient_optimisation'):
        gradient_optimizer_op = tf.train.AdamOptimizer(param_dict['learning_rate'])
        gd_opt_op = gradient_optimizer_op.minimize(loss_op)
    return loss_op, gd_opt_op

def accuracy(predictions, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels,1))
        accuracy = 100*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def mk_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def log_file_suffix(params):
    file_suffix = ""
    for elem in params:
        file_suffix = file_suffix + "_" + str(elem)
    return file_suffix

def fitness(param_space_values):
    # Here we re-create the param dictionary
    project_path = os.getcwd()
    param_dict = {}
    for i in range(len(param_space_values)):
        param_dict[default_param_name[i]] = param_space_values[i]
    # Rest of the parameter lies in param.txt
    param_dict_original = get_hyper_parameters(args.param, project_path, False)
    for key, value in param_dict_original.items():
        if key not in param_dict.keys():
            param_dict[key] = value
    ########### param_dict creation complete ################
    # Define the data provider module
    img_batch, label_batch, dp = get_data_provider(args.dataset, project_path, param_dict)
    dp.set_batch(param_dict['BATCH_SIZE'])
    # Construct a model to be trained
    model = get_model(args.model, img_batch, param_dict)
    # Define optimization procedure
    logits = model['feature_logits']
    output_probability = model['feature_out']
    loss_op, gd_opt_op = optimisation(label_batch, logits, param_dict)
    # Define accuracy operation
    accuracy_op = accuracy(output_probability, label_batch)
    # Merge all summaries
    summary_op = tf.summary.merge_all()
    summary_path = project_path + "/debug/opt_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    # Start a session
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_path + '/' + log_file_suffix(param_space_values), sess.graph)
        sess.run(tf.global_variables_initializer())
        for nr_epochs in range(param_dict['NUM_EPOCHS']):
            for i in range(5):
                img_batch_data, label_batch_data = dp.next()
                feed_dict = {img_batch: img_batch_data, label_batch: label_batch_data}
                _, out, loss, accu, summary = sess.run([gd_opt_op,output_probability,loss_op, accuracy_op, summary_op], feed_dict = feed_dict)
                train_writer.add_summary(summary, nr_epochs*10 + i)
        train_writer.close()
    tf.reset_default_graph()
    print(accu)
    return -accu

def execute(arguments):
    global args
    args = arguments
    project_path = os.getcwd()
    param_dict, param_dict_space = get_hyper_parameters(args.param, project_path)
    sanity_check(param_dict, param_dict_space)
    if not default_param_name:
        print("Fix your param, exiting")
        sys.exit()
    search_result = gp_minimize(func=fitness,
                            dimensions=space_dimensions,
                            acq_func='EI',
                            n_calls=20,
                            x0=default_param_value)
    print("Best parameters has been searched")
    best_results = search_result.x
    param_dict_opt = param_dict
    for i in range(len(default_param_name)):
        param_dict_opt[default_param_name[i]] = best_results[i]
    set_hyper_parameter(args.param, project_path, param_dict_opt)
