"""
Dataset script for CIFAR-10
Used tensorflow TFRecords for datapipeline
"""

import wget
import tarfile
import cPickle
import tensorflow as tf
import numpy as np
import cv2
import sys
import os

class CIFAR_10:
    def __init__(self, absolute_path):
        self.dataset_name = "CIFAR-10"
        self.absolute_path = absolute_path

    def get_data(self):
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        dataset_path = self.absolute_path + "/dataset/" + self.dataset_name
        if not os.path.isdir(dataset_path + "/raw"):
            os.mkdir(dataset_path + "/raw")
        self.filename = wget.download(self.url, dataset_path + "/raw")
        print("Downloaded "+ self.filename)
        tar = tarfile.open(self.filename)
        tar.extractall(path = dataset_path +"/raw")
        tar.close()
        print("Extracted the raw data.")

    def preprocess_data(self):
        raw_folder = "cifar-10-batches-py"
        dataset_path = self.absolute_path + "/dataset/" + self.dataset_name
        out_file = dataset_path + "/pre-processed/" + "dataset.tfrecords"
        if not os.path.isdir(dataset_path + "/pre-processed"):
            os.mkdir(dataset_path + "/pre-processed")
        writer = tf.python_io.TFRecordWriter(out_file)
        for i in range(1,6):
            filename = dataset_path + "/raw/" + raw_folder+"/data_batch_"+str(i)
            f = open(filename, 'rb')
            dict = cPickle.load(f)
            images = dict['data']
            images = np.reshape(images, (10000, 3, 32, 32))
            labels = dict['labels']
            for j in range(0,10000):
                img = np.transpose(images[j], (1,2,0))
                img = self.process_image(img)
                label = labels[j]
                example = tf.train.Example(
                        features=tf.train.Features(
                            feature = {
                                'img': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                                         value=[img.tostring()])
                                ),
                                'label': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[label])
                                )
                            }
                        )
                )
                writer.write(example.SerializeToString())
            f.close()
        writer.close()
        sys.stdout.flush()
        print("Images are pre-processed.")

    def process_image(self, img):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        return img

def get_obj(absolute_path):
    obj = CIFAR_10(absolute_path)
    return obj
