"""
Data provider for IMAGENET
This file is based on source code provided by tensorflow/model/inception
Imagenet dataset, once preprocessed with the preprocessing script, creates
tfrecord file splitted into multiple files having (1024 files) and is kept at
preprocessing directory. The file pattern is train-0001-of-01024 or %s-*
We use tf.slim tfexample_decoder to decode the tfrecorder file and dataset to
read all the tfrecoder files in train or test directory.
slim also provide dataset_data_provider which we use to fetch a batch of data
for training or testing
"""

import tensorflow as tf
import cv2
import numpy as np
import os
slim = tf.contrib.slim

# below utility variables are required to use tf.slim for data provider
#------------------------------------------------------------------------------|
_FILE_PATTERN = '%s-*'
_SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}
_NUM_CLASSES = 1001
keys_to_features = {
  'image/encoded': tf.FixedLenFeature(
      (), tf.string, default_value=''),
  'image/format': tf.FixedLenFeature(
      (), tf.string, default_value='jpeg'),
  'image/class/label': tf.FixedLenFeature(
      [], dtype=tf.int64, default_value=-1),
  'image/class/text': tf.FixedLenFeature(
      [], dtype=tf.string, default_value=''),
  'image/object/bbox/xmin': tf.VarLenFeature(
      dtype=tf.float32),
  'image/object/bbox/ymin': tf.VarLenFeature(
      dtype=tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(
      dtype=tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(
      dtype=tf.float32),
  'image/object/class/label': tf.VarLenFeature(
      dtype=tf.int64),
}
items_to_handlers = {
  'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
  'label': slim.tfexample_decoder.Tensor('image/class/label'),
  'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
  'object/bbox': slim.tfexample_decoder.BoundingBox(
      ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
  'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
}
#------------------------------------------------------------------------------|

#TODO: Create a different module called data_augmentation
# For now, augmentation module can be a part of data provider
def preprocess_for_train(img, h, w):
    image = tf.image.convert_image_dtype(img, dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image,[h, w],align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def preprocess_for_eval(img, h, w):
    return preprocess_for_train(img,h,w)
#------------------------------------------------------------------------------|

class data_provider_IMAGENET:
    def __init__(self, absolute_path):
        self.dataset_name = "IMAGENET"
        self.absolute_path = absolute_path
        self.img_size = 224

    def train_data_pipeline(self):
        self.decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                              items_to_handlers)
        self.reader = tf.TFRecordReader
        tfrecord_file_path = self.absolute_path+"/dataset/" +self.dataset_name +
                            "/pre-processed/tfrecords"
        file_pattern = os.path.join(self.dataset_dir, _FILE_PATTERN % "train")
        self.slim_dataset = slim.dataset.Dataset(data_sources = file_pattern,
                            reader = self.reader, decoder=self.decoder,
                            num_samples = _SPLITS_TO_SIZES['train'],
                            items_to_descriptions = _ITEMS_TO_DESCRIPTIONS,
                            num_classes = _NUM_CLASSES, labels_to_names = None)
        dp = slim.dataset_data_provider.DatasetDataProvider(self.slim_dataset,
                num_readers = 1, common_queue_capacity = 20*self.batch_size,
                common_queue_min = 10*self.batch_size)
        [self.unpreprocessed_image, self.label] = dp.get(['image', 'label'])
        self.processed_image =preprocessing_for_train(self.unpreprocessed_image,
                                self.img_size, self.img_size)
        return




    # For imagenet training, this function call in essential
    def set_batch(self, batch_size):
        self.batch_size = batch_size
        self.img_batch, self.label_batch = self.data_provider()

    def next(self):
        with tf.train.MonitoredSession() as sess:
            img, label = sess.run([self.img_batch, self.label_batch])
        return [img, label]

    def view(self, batch_data):
        img = batch_data[0]
        label = batch_data[1]
        print(label[0])
        print(img[0].shape)
        img_path = self.absolute_path + "/debug/image.jpg"
        cv2.imwrite(img_path, img[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def data_provider(self):
        # This is where we scale the image in [-1, 1] range
        #self.image = tf.image.per_image_standardization(self.processed_image)
        # image standardization transforms every image to have mean 0 and std 1
        self.train_data_pipeline()
        images, labels = tf.train.batch([self.processed_image, self.label],
                                  batch_size=self.batch_size,
                                  num_threads=4, allow_smaller_final_batch=True,
                                  capacity=5*self.batch_size)
        return images, labels

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
