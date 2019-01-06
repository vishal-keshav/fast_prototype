"""
Data provider for IMAGENET
"""

import tensorflow as tf
import cv2
import numpy as np

#TODO: complete this
class data_provider_IMAGENET:
    def __init__(self, absolute_path):
        self.dataset_name = "MNIST"
        self.absolute_path = absolute_path

    def set_batch(self, batch_size):
        pass

    def next(self):
        pass

    def validation(self):
        pass

    def view(self, batch_data):
        img = batch_data[0]
        label = batch_data[1]
        print(label[0])
        print(img[0].shape)
        img_path = self.absolute_path + "/debug/image.jpg"
        cv2.imwrite(img_path, img[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def get_obj(absolute_path):
    obj = data_provider_IMAGENET(absolute_path)
    return obj
