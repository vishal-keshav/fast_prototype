"""
Data provider for IMAGENET

"""

import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
import os

class data_provider_imagenet2012:
    """
    ImageNet dataset from tensorflow_datasets
    """
    def __init__(self, absolute_path):
        """
        Create dataset for train and validation
        """
        self.dataset_name = "imagenet2012"
        self.absolute_path = absolute_path
        self.one_shot = False
        self.distributed = False
        #self.dataset_dir = absolute_path+"/dataset/"+ self.dataset_name +\
        # "/pre-processed"
        #self.raw_dir = absolute_path+"/dataset/"+ self.dataset_name +\
        # "/raw"
        self.raw_dir = "/media/bulletcross/VK_SSD/dataset"
        self.dataset_dir = "/media/bulletcross/VK_SSD/imagenet_preprocessed"
        self.imagenet_builder = tfds.builder(self.dataset_name,
                                                    data_dir = self.dataset_dir)
        #self.imagenet_builder.download_and_prepare(download_dir = self.raw_dir)
        self.imagenet_builder.download_and_prepare()
        self.train_dataset = self.imagenet_builder.as_dataset(
                                            split = tfds.Split.TRAIN)
        self.validation_dataset = self.imagenet_builder.as_dataset(
                                            split = tfds.Split.VALIDATION)

    def _img_transform(self, img):
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)
        return img

    def train_transform(self, input):
        """
        Dataset preprocessing for train dataset
        TODO: Add data augmentation
        """
        img = tf.image.convert_image_dtype(image = input["image"],
                                                    dtype = tf.float32)
        img = tf.image.resize_images(images = img, size = [224, 224])
        img = self._img_transform(img)
        label = tf.one_hot(indices = input["label"]+1, depth=1001, axis=-1)
        return {"image": img, "label": label}

    def validation_transform(self, input):
        """
        Dataset preprocessing for validation dataset
        """
        img = tf.image.convert_image_dtype(image = input["image"],
                                                    dtype = tf.float32)
        img = tf.image.resize_images(images = img, size = [224, 224])
        img = self._img_transform(img)
        label = tf.one_hot(indices = input["label"]+1, depth=1001, axis=-1)
        return {"image": img, "label": label}

    def get_train_dataset(self, batch_size = 128, shuffle = 1, prefetch = 1):
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
        self.train_dataset = self.train_dataset.shuffle(shuffle)
        self.train_dataset = self.train_dataset.prefetch(prefetch)
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

    def get_validation_dataset(self, batch_size = 128, shuffle =1, prefetch =1):
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
        self.validation_dataset = self.validation_dataset.shuffle(shuffle)
        self.validation_dataset = self.validation_dataset.prefetch(prefetch)
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
        shape_dict = {'input': (None, 224, 224, 3), 'output': (None, 1001)}
        return shape_dict

def get_obj(absolute_path):
    obj = data_provider_imagenet2012(absolute_path)
    return obj

#----------------------------Testing-----------------------------

def main():
    project_path = os.getcwd()
    dp_obj = get_obj(project_path)
    img, label = dp_obj.get_train_dataset(batch_size = 1)
    with tf.Session() as sess:
        dp_obj.initialize_train(sess)
        im, lab = sess.run([img, label])
        print(im.shape)
        print(lab)

if __name__ == "__main__":
    main()
