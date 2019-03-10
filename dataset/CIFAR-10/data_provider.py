"""
Data provider for CIFAR-10
"""

import tensorflow as tf
import cv2

class data_provider_CIFAR_10:
    def __init__(self, absolute_path):
        self.dataset_name = "CIFAR-10"
        self.absolute_path = absolute_path

    def _tfrecord_parser(self, record_file_path):
        parse_op = tf.parse_single_example(serialized = record_file_path,
                    features = {
                        'img': tf.FixedLenFeature([], tf.string),
                        'label'  : tf.FixedLenFeature([], tf.int64)
                    },
                    name = 'parse_op')
        img = tf.decode_raw(parse_op['img'], tf.uint8, name = 'byte_to_int8_op')
        img = tf.cast(img, tf.float32, name = 'int8_to_int32_op')
        img = tf.reshape(img, shape=[32, 32, 3], name = 'reshape_op')
        label = tf.cast(parse_op['label'], tf.int32, name = 'label_cast_op')
        label = tf.one_hot(indices = label, depth = 10, axis=-1)
        return img, label

    def _get_tfrecord_dataset_train(self,tfrecord_datafile):
        dataset = tf.data.TFRecordDataset(filenames = tfrecord_datafile)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1024, 1))
        dataset = dataset.apply(
           tf.contrib.data.map_and_batch(self._tfrecord_parser,self.batch_size))
        dataset =dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
        return dataset

    def _get_tfrecord_dataset_valid(self,tfrecord_datafile):
        dataset = tf.data.TFRecordDataset(filenames = tfrecord_datafile)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1024, 1))
        dataset = dataset.apply(tf.contrib.data.map_and_batch(self._tfrecord_parser, 10000))
        dataset =dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
        return dataset

    def _initialize_initializer(self):
        train_file_name = self.absolute_path + "/dataset/"+self.dataset_name + \
                                         "/pre-processed/" + "dataset.tfrecords"
        val_file_name = self.absolute_path + "/dataset/"+self.dataset_name + \
                                    "/pre-processed/" + "dataset_val.tfrecords"
        self.train_dataset = self._get_tfrecord_dataset_train(train_file_name)
        self.validation_dataset =self._get_tfrecord_dataset_valid(val_file_name)
        self.iterator = tf.data.Iterator.from_structure(
               self.train_dataset.output_types,self.train_dataset.output_shapes)
        self.next_element = self.iterator.get_next()
        self.training_init_op=self.iterator.make_initializer(self.train_dataset)
        self.validation_init_op = self.iterator.make_initializer(
                                                self.validation_dataset)

    def get_input_output(self):
        self._initialize_initializer()
        return self.next_element

    def get_dataset_shape(self):
        input_shape = np.array([None, 32, 32, 3])
        output_shape = np.array([None, 10])
        return input_shape, output_shape

    def set_train_batch(self, batch_size):
        self.batch_size = batch_size
        self.train_size = 50000
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
    obj = data_provider_CIFAR_10(absolute_path)
    return obj
