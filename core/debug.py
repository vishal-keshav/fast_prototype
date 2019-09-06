"""
Debug dataset is a module for debugging problems arising from
wrong collection of the dataset. The analysis of dataset problems can
only be identified if developers looks into two things:
1. How does sample inputs looks like.
2. What is the input and output distributions. Are there any imbalance?

Here, we assume datasets as our inherent module, providing visual inputs
and class labels.

This is an independent module, and later will be included through debug core.
"""
import os
import importlib
import numpy as np
import cv2
import scipy
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import rawpy
import time

import tensorflow as tf

def get_hyper_parameters(version, project_path):
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(version, project_path)
    param_dict = hp_obj.get_params()
    return param_dict

def get_data_provider(dataset, project_path, param_dict):
    data_provider_module_path = "dataset." + dataset + ".data_provider"
    data_provider_module = importlib.import_module(data_provider_module_path)
    dp_obj = data_provider_module.get_obj(project_path)
    return dp_obj

def get_model(version, inputs, param_dict, is_train = False, pretrained =False):
    model_module_path = "architecture.version" + str(version) + ".model"
    model_module = importlib.import_module(model_module_path)
    model = model_module.create_model(inputs, is_train, pretrained)
    return model

def optimisation(label_batch, logits, param_dict):
    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=label_batch, logits=logits))
    tf.summary.scalar('loss', loss_op)
    with tf.name_scope('gradient_optimisation'):
        gradient_opt_op = tf.train.AdamOptimizer(param_dict['learning_rate'])
        gd_opt_op = gradient_opt_op.minimize(loss_op)
    return loss_op, gd_opt_op

def accuracy(predictions, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1),
                                      tf.argmax(labels,1))
        accuracy = 100.0*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def mk_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

class distribution_plotter:
    def __init__(self):
        self.distribution_data = {}

    def add_distribution_data(self, data, name):
        if name in self.distribution_data.keys():
            self.distribution_data[name] = \
                            np.append(self.distribution_data[name], data)
        else:
            self.distribution_data[name] = data

    def reset(self):
        self.distribution_data.clear()
        plt.clf()

    def plot_distribution(self, title = 'plot', xlabel = 'xlable',
                                ylabel = 'ylabel', path = ""):
        for key, value in self.distribution_data.items():
            sns.distplot(value, hist = False, kde = True,
                        kde_kws = {'linewidth': 0.5}, label = key)
        plt.figure(1, figsize=(1000,1000))
        #plt.legend(loc=1, prop={'size': 3}, title = title)
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left",
                                        prop={'size': 3}, borderaxespad=0)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(path, title + '.jpg'), dpi=600)
#-------------------------------------------------------------------------------

def dataset_input_samples(dp, nr_samples = 10):
    dp.make_one_shot
    train_img, train_label = dp.get_train_dataset(batch_size = nr_samples,
                                                    shuffle = 1, prefetch = 1)
    project_path = os.getcwd()
    write_path = os.path.join(project_path, "debug", dp.dataset_name)
    mk_dir(write_path)
    dist_plot = distribution_plotter()

    with tf.Session() as sess:
        dp.initialize_train(sess)
        img,label = sess.run([train_img, train_label])
        for i in range(nr_samples):
            scipy.misc.imsave(os.path.join(
                                write_path,"file"+str(i)+".jpg"),img[i])
            dist_plot.add_distribution_data(img[i].flatten(), "input_distr")
            dist_plot.plot_distribution("file_dist"+str(i), path = write_path)
            dist_plot.reset()
            print(label[i])

def dataset_label_distribution(dp, dataset_type = "validation"):
    dp.make_one_shot
    project_path = os.getcwd()
    write_path = os.path.join(project_path, "debug", dp.dataset_name)
    mk_dir(write_path)
    if dataset_type == "validation":
        img, label = dp.get_validation_dataset(batch_size = 256,
                                                    shuffle = 1, prefetch = 1)
    else:
        img, label = dp.get_train_dataset(batch_size = 256,
                                                    shuffle = 1, prefetch = 1)
    nr_classes = dp.get_input_output_shape()['output'][1]
    label_distribution = np.zeros(nr_classes)
    index = np.arange(len(label_distribution))
    with tf.Session() as sess:
        dp.initialize_validation(sess)
        while True:
            try:
                label = sess.run(label)
                for i in range(label.shape[0]):
                    label_distribution[np.argmax(label[i])] = \
                            label_distribution[np.argmax(label[i])] + 1
            except tf.errors.OutOfRangeError:
                print("completed")
                break
    plt.bar(index, label_distribution)
    plt.title("Validation data class distribution")
    plt.savefig(os.path.join(write_path, 'val_dist.jpg'), dpi=600)

def train_overfit(model, dp, param_dict, nr_samples = 64):
    # Prepare a batch of data, on which overfitting should happen
    train_img, train_label = dp.get_train_dataset(batch_size = nr_samples,
                                                    shuffle = 1, prefetch = 1)
    label = tf.placeholder(tf.float32, dp.get_input_output_shape()['output'])
    loss_op, gd_opt_op = optimisation(label, model['feature_logits'],param_dict)
    accuracy_op = accuracy(model['feature_out'], label)
    with tf.Session() as sess:
        # create a batch dataset
        dp.initialize_train(sess)
        input_batch, label_batch = sess.run([train_img, train_label])
        assert input_batch.shape[0] == nr_samples
        assert label_batch.shape[0] == nr_samples
        # train, over fit and achive 100% accuracy
        sess.run(tf.initializers.global_variables())
        while True:
            _, loss, acc = sess.run([gd_opt_op, loss_op, accuracy_op],
             feed_dict = {model['feature_in']: input_batch, label: label_batch})
            print(loss, acc)

def ltsid_overfit(model):
    label = tf.placeholder(tf.float32, [None, None, None, 3])
    loss_op = tf.reduce_mean(tf.abs(model['feature_out']-label))
    opt_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_op)
    input_file_name = "exposure_0.1.dng"
    gt_file_name = "exposure_10.dng"
    model_input = rawpy.imread(input_file_name)
    model_input = model_input.raw_image_visible.astype(np.float32)
    gt_label = rawpy.imread(gt_file_name).postprocess(use_camera_wb=True,
                            half_size=False, no_auto_bright=True, output_bps=16)
    gt_label = np.expand_dims(np.float32(gt_label / 65535.0), axis=0)
    scipy.misc.imsave(os.path.join("gt_preprocessed.jpg"), gt_label[0])
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        for i in range(1000):
            _, loss, out = sess.run([opt_op, loss_op, model['feature_out']],
                feed_dict = {model['feature_in']: model_input, label: gt_label})
            print(loss)
        scipy.misc.imsave(os.path.join("reconstructed.jpg"), out[0])

def execute(args):
    tf.random.set_random_seed(1234)
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(args.param, project_path)
    #dp = get_data_provider(args.dataset, project_path, param_dict)
    #dataset_input_samples(dp)
    #dataset_label_distribution(dp)
    #train_overfit(model, dp, param_dict)
    #debug_distributed_training(args.model, args.dataset, param_dict)
    #input = tf.placeholder(tf.float32, dp.get_input_output_shape()['input'])
    #input = tf.placeholder(tf.float32, shape = [None, None])
    #model = get_model(args.model, input, param_dict, is_train = True,
    #                                                        pretrained = False)
    #ltsid_overfit(model)
