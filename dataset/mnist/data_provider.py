"""
Data provider for MNIST
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np

import os

class data_provider_mnist:
    """
    Mnist dataset from tensorflow_datasets
    """
    def __init__(self, absolute_path):
        """
        Create dataset for train and test
        """
        self.dataset_name = "mnist"
        self.absolute_path = absolute_path
        self.one_shot = False
        self.distributed = False
        #raw_path = self.absolute_path +"/dataset/" + self.dataset_name + "/raw"
        #pre_processed_path =self.absolute_path +"/dataset/" +self.dataset_name\
        #                                                      +"/pre-processed"
        raw_path = "/media/bulletcross/VK_SSD/mnist_dataset"
        pre_processed_path = "/media/bulletcross/VK_SSD/mnist_preprocessed"
        self.mnist_builder = tfds.builder(self.dataset_name,
                                                data_dir = pre_processed_path)
        self.mnist_builder.download_and_prepare(download_dir = raw_path)
        self.train_dataset = self.mnist_builder.as_dataset(
                                                       split = tfds.Split.TRAIN)
        self.validation_dataset = self.mnist_builder.as_dataset(
                                                        split = tfds.Split.TEST)

    def train_transform(self, input):
        img = tf.image.convert_image_dtype(image = input["image"], dtype = tf.float32)
        img = tf.image.resize(images = img, size = [28, 28])
        label = tf.one_hot(input["label"], depth = 10, axis = -1)
        return {"image": img, "label": label}

    def validation_transform(self, input):
        img = tf.image.convert_image_dtype(image = input["image"], dtype = tf.float32)
        img = tf.image.resize(images = img, size = [28, 28])
        label = tf.one_hot(input["label"], depth = 10, axis = -1)
        return {"image": img, "label": label}

    def get_train_dataset(self, batch_size = 128, shuffle = 1024, prefetch = 1):
        """
        Applies transformation on the dataset and return tensor iterator element

        Args:
            batch_size: Number of element to process at once.
            shuffle: Number of elements to shuffle before returning data
            prefetch: Number of databatch to keep in memory

        Returns:
            dataset iterator element for input and label in a disctionary
            keyed by 'image' and 'label'
        """
        self.train_dataset = self.train_dataset.prefetch(prefetch)
        self.train_dataset = self.train_dataset.shuffle(shuffle)
        self.train_dataset = self.train_dataset.map(self.train_transform)
        self.train_dataset = self.train_dataset.batch(batch_size)
        if self.one_shot:
            self.iter_train = self.train_dataset.make_one_shot_iterator()
        elif self.distributed:
            self.iter_train = self.strategy.make_dataset_iterator(
                                                      self.train_dataset)
            return self.iter_train
        else:
            self.iter_train = self.train_dataset.make_initializable_iterator()
        train_elem = self.iter_train.get_next()
        return train_elem["image"], train_elem["label"]

    def get_validation_dataset(self,batch_size = 1, shuffle = 1, prefetch = 1):
        """
        Applies transformation on the dataset and return tensor iterator element

        Args:
            batch_size: Number of element to process at once.
            shuffle: Number of elements to shuffle before returning data
            prefetch: Number of databatch to keep in memory

        Returns:
            dataset iterator element for input and label in a disctionary
            keyed by 'image' and 'label'
        """
        self.validation_dataset = self.validation_dataset.prefetch(prefetch)
        self.validation_dataset = self.validation_dataset.shuffle(1024)
        self.validation_dataset = self.validation_dataset.map(
                                                self.validation_transform)
        self.validation_dataset = self.validation_dataset.batch(batch_size)
        if self.one_shot:
            self.iter_valid = self.validation_dataset.make_one_shot_iterator()
        elif self.distributed:
            self.iter_valid = self.strategy.make_dataset_iterator(
                                                      self.validation_dataset)
            return self.iter_valid
        else:
            self.iter_valid = \
                          self.validation_dataset.make_initializable_iterator()
        validation_elem = self.iter_valid.get_next()
        return validation_elem["image"], validation_elem["label"]

    def make_one_shot(self):
        self.one_shot = True

    def make_distributed(self, strategy):
        self.distributed = True
        self.strategy = strategy

    def initialize_train(self, sess):
        if self.distributed:
            sess.run(self.iter_train.initialize())
            return
        sess.run(self.iter_train.initializer)

    def initialize_validation(self, sess):
        if self.distributed:
            sess.run(self.iter_valid.initialize())
            return
        sess.run(self.iter_valid.initializer)

    def get_input_output_shape(self):
        """
        Returns the shapes of input and output used by default.
        Returns:
            Dictionary keyed by 'input' and 'output'
        """
        shape_dict = {'input': (None, 28, 28, 1), 'output': (None, 10)}
        return shape_dict

def get_obj(absolute_path):
    obj = data_provider_mnist(absolute_path)
    return obj

#-----------Test----------------------

def main():
    path = os.getcwd()
    obj = get_obj(path)
    x,y = obj.get_validation_dataset()
    print(x,y)

if __name__ == "__main__":
    main()
