"""
Data provider for COCO 2014
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np

import os

class data_provider_coco:
    """
    Coco2014 dataset from tensorflow_datasets
    """
    def __init__(self, absolute_path):
        """
        Create dataset for train and test
        """
        self.dataset_name = "coco2014"
        self.absolute_path = absolute_path
        self.one_shot = False
        self.distributed = False
        #raw_path = self.absolute_path +"/dataset/" + self.dataset_name + "/raw"
        #pre_processed_path =self.absolute_path +"/dataset/" +self.dataset_name\
        #                                                      +"/pre-processed"
        raw_path = "/media/bulletcross/VK_SSD/coco2014_dataset"
        pre_processed_path = "/media/bulletcross/VK_SSD/coco2014_preprocessed"
        self.coco_builder = tfds.builder(self.dataset_name,
                                                data_dir = pre_processed_path)
        self.coco_builder.download_and_prepare(download_dir = raw_path)
        self.train_dataset = self.coco_builder.as_dataset(
                                                       split = tfds.Split.TRAIN)
        self.validation_dataset = self.coco_builder.as_dataset(
                                                  split = tfds.Split.VALIDATION)

    def train_transform(self, input):
        img = tf.image.convert_image_dtype(image = input["image"], dtype = tf.float32)
        img = tf.image.resize(images = img, size = [416, 416])
        bbox = input["objects"]["bbox"]
        label = tf.one_hot(input["objects"]["label"], depth = 80, axis = -1)
        return {"image": img, "bbox": bbox, "label": label}

    def validation_transform(self, input):
        img = tf.image.convert_image_dtype(image = input["image"], dtype = tf.float32)
        img = tf.image.resize(images = img, size = [416, 416])
        bbox = input["objects"]["bbox"]
        label = tf.one_hot(input["objects"]["label"], depth = 80, axis = -1)
        return {"image": img, "bbox": bbox, "label": label}

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
        return train_elem["image"], train_elem["bbox"], train_elem["label"]

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
        return validation_elem["image"], validation_elem["bbox"], \
                                                    validation_elem["label"]

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
        shape_dict = {'input': (None, 416, 416, 3), 'output': (None, 80)}
        return shape_dict

def get_obj(absolute_path):
    obj = data_provider_coco(absolute_path)
    return obj

#-----------Test----------------------

def main():
    path = os.getcwd()
    obj = get_obj(path)
    x,y,z = obj.get_train_dataset()
    with tf.Session() as sess:
        obj.initialize_train(sess)
        x_out, y_out, z_out = sess.run([x,y,z])
    print(x_out)
    print(y_out)
    print(y_out.shape)
    print(z_out.shape)

if __name__ == "__main__":
    main()
