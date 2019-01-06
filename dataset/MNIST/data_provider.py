"""
Data provider for MNIST
"""

import tensorflow as tf
import cv2
import numpy as np

class data_provider_MNIST:
    def __init__(self, absolute_path):
        self.dataset_name = "MNIST"
        self.absolute_path = absolute_path
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
        # for validation (for now, just for defining the API, we treat test set as validation set)
        test_x, test_y = self.mnist.test.images, self.mnist.test.labels
        self.validation_x = np.array(test_x).reshape((-1, 28, 28, 1))
        self.validation_y = np.array(test_y)

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def next(self):
        batch_x_list, batch_y = self.mnist.train.next_batch(self.batch_size)
        batch_x = np.array(batch_x_list).reshape((-1, 28, 28, 1))
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def validation(self):
        return self.validation_x, self.validation_y

    def view(self, batch_data):
        img = batch_data[0]
        label = batch_data[1]
        print(label[0])
        print(img[0].shape)
        img_path = self.absolute_path + "/debug/image.jpg"
        cv2.imwrite(img_path, img[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def get_obj(absolute_path):
    obj = data_provider_MNIST(absolute_path)
    return obj
