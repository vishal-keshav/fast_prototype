"""
Data provider for MNIST
"""

import tensorflow as tf
import cv2
import numpy as np

import os

class data_provider_MNIST:
    def __init__(self, absolute_path):
        self.dataset_name = "MNIST"
        self.absolute_path = absolute_path
        ########################################################################
        from tensorflow.examples.tutorials.mnist import input_data
        raw_path = self.absolute_path + "/dataset/" + self.dataset_name + "/raw"
        pre_processed_path =self.absolute_path + "/dataset/" +self.dataset_name\
                                                              + "/pre-processed"
        self.mnist = input_data.read_data_sets(raw_path, one_hot = True)

    def _initialize_initializer(self):
        # here we create train, validation and test datasets
        train_x = self.mnist.train.images
        train_y = self.mnist.train.labels
        train_x_reshaped = np.array(train_x).reshape((train_x.shape[0], 28, 28, 1))
        train_y_reshaped = np.array(train_y).reshape((train_y.shape[0], 10))
        self.train_dataset = (tf.data.Dataset.from_tensor_slices(
                                            (train_x_reshaped,
                                             train_y_reshaped))
                                            .repeat()
                                            .batch(self.batch_size))
        test_x = self.mnist.test.images
        test_y = self.mnist.test.labels
        test_x_reshaped = np.array(test_x).reshape((test_x.shape[0], 28, 28, 1))
        test_y_reshaped = np.array(test_y).reshape((test_x.shape[0], 10))
        self.validation_dataset = (tf.data.Dataset.from_tensor_slices(
                                            (test_x_reshaped,
                                             test_y_reshaped))
                                            .repeat()
                                            .batch(test_x.shape[0]))
        # here we create a generic iterator (used for creating the network)
        self.iterator = tf.data.Iterator.from_structure(
               self.train_dataset.output_types,self.train_dataset.output_shapes)
        self.next_element = self.iterator.get_next()
        # here we create train, validation and test initializers
        self.training_init_op=self.iterator.make_initializer(self.train_dataset)
        self.validation_init_op = self.iterator.make_initializer(
                                                self.validation_dataset)

    def get_dataset_shape(self):
        input_shape = np.array([None, 28, 28, 1])
        output_shape = np.array([None, 10])
        return input_shape, output_shape

    def get_input_output(self):
        self._initialize_initializer()
        return self.next_element

    def set_train_batch(self, batch_size):
        self.batch_size = batch_size
        self.train_size = 60000
        self.nr_batch = int(self.train_size/self.batch_size)

    def get_nr_train_batch(self):
        return self.nr_batch

    def set_for_train(self, sess):
        sess.run(self.training_init_op)

    def set_for_validation(self, sess):
        sess.run(self.validation_init_op)

    def set_for_testing(self, sess):
        sess.run(self.validation_init_op)

def get_obj(absolute_path):
    obj = data_provider_MNIST(absolute_path)
    return obj
