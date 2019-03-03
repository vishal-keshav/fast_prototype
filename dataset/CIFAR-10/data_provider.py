"""
Data provider for CIFAR-10
"""

import tensorflow as tf
import cv2

class data_provider_CIFAR_10:
    def __init__(self, absolute_path):
        self.dataset_name = "CIFAR-10"
        self.absolute_path = absolute_path

    def tfrecord_parser(self, record_file_path):
        parse_op = tf.parse_single_example(serialized = record_file_path,
                    features = {
                        'img': tf.FixedLenFeature([], tf.string),
                        'label'  : tf.FixedLenFeature([], tf.int64)
                    },
                    name = 'parse_op')
        img = tf.decode_raw(parse_op['img'], tf.uint8, name = 'byte_to_int8_op')
        img = tf.cast(img, tf.float32, name = 'int8_to_int32_op')
        img = tf.reshape(img, shape=[224, 224, 3], name = 'reshape_op')
        label = tf.cast(parse_op['label'], tf.int32, name = 'label_cast_op')
        label = tf.one_hot(indices = label, depth = 10, axis=-1)
        return img, label

    def train_data_pipeline(self, train_data_file, PREFETCH_BUFFER, NUM_EPOCHS,
                                                       BATCH_SIZE, GPU_ENABLED):
        dataset = tf.data.TFRecordDataset(filenames = train_data_file,
                                            num_parallel_reads = 8)
        dataset = dataset.apply(
                    tf.contrib.data.shuffle_and_repeat(PREFETCH_BUFFER,
                                                       NUM_EPOCHS))
        dataset = dataset.apply(
                    tf.contrib.data.map_and_batch(self.tfrecord_parser,
                                                  BATCH_SIZE))
        if GPU_ENABLED:
            dataset =dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
        return dataset

    def data_provider(self, args):
        file_name = self.absolute_path + "/dataset/" + self.dataset_name +
                                         "/pre-processed/" + "dataset.tfrecords"
        dataset = self.train_data_pipeline(file_name, args['PREFETCH_BUFFER'],
                    args['NUM_EPOCHS'], args['BATCH_SIZE'], args['GPU_ENABLED'])
        iterator = dataset.make_one_shot_iterator()
        img_batch, label_batch = iterator.get_next()
        return img_batch, label_batch

    def set_batch(self, batch_size):
        self.batch_size = batch_size
        args = {'PREFETCH_BUFFER': 1024,
                'NUM_EPOCHS': 1,
                'BATCH_SIZE': batch_size,
                'GPU_ENABLED': False}
        self.img_batch, self.label_batch = self.data_provider(args)

    def next(self):
        with tf.Session() as sess:
            img, label = sess.run([self.img_batch, self.label_batch])
        return [img, label]

    def view(self, batch_data):
        img = batch_data[0]
        label = batch_data[1]
        print(label[0])
        print(img[0].shape)
        img_path = self.absolute_path + "/debug/image.jpg"
        cv2.imwrite(img_path, img[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def get_obj(absolute_path):
    obj = data_provider_CIFAR_10(absolute_path)
    return obj
