"""
Dataset script for COCO 2014
"""

import wget
import tarfile
import six
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

class coco2014:
    def __init__(self, absolute_path):
        self.dataset_name = "coco2014"
        self.absolute_path = absolute_path

    def get_data(self):
        dataset_path = self.absolute_path + "/dataset/" + self.dataset_name
        if not os.path.isdir(dataset_path + "/raw"):
            os.mkdir(dataset_path + "/raw")

    def preprocess_data(self):
        dataset_path = self.absolute_path + "/dataset/" + self.dataset_name
        if not os.path.isdir(dataset_path + "/pre-processed"):
            os.mkdir(dataset_path + "/pre-processed")

def get_obj(absolute_path):
    obj = coco2014(absolute_path)
    return obj
