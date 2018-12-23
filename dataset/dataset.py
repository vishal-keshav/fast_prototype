# Dataset script for CIFAR-10
import wget
import tarfile
import cPickle
import tensorflow as tf
import numpy as np
import cv2
import sys

#absolute_path = "/home/vishal/ml_prototype/"

class DataSet:
    def __init__(self, dataset_name, absolute_path):
        self.dataset_name = dataset_name
        self.absolute_path = absolute_path

    def get_data(self):
        self.url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.filename = wget.download(self.url, self.absolute_path + "/dataset/" + self.dataset_name + "/raw")
        print("Downloaded "+ self.filename)
        tar = tarfile.open(self.absolute_path, "/dataset/" + self.dataset_name + "/raw/" + self.filename)
        #tar = tarfile.open(self.absolute_path + "dataset/" + self.dataset_name + "/raw/" + name)
        tar.extractall(path = self.absolute_path + "/dataset/" + self.dataset_name +"/raw")
        tar.close()
        print("Extracted the raw data.")

    def preprocess_data(self):
        raw_folder = "cifar-10-batches-py"
        out_file = self.absolute_path + "/dataset/" + self.dataset_name + "/pre-processed/" + "dataset.tfrecords"
        writer = tf.python_io.TFRecordWriter(out_file)
        for i in range(1,6):
            filename = self.absolute_path + "/dataset/" + self.dataset_name + "/raw/" + raw_folder + "/data_batch_" + str(i)
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
                                    bytes_list=tf.train.BytesList(value=[img.tostring()])
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

    def tfrecord_parser(self, record_file_path):
        parse_op = tf.parse_single_example(serialized = record_file_path,
                    features = {
                        'img': tf.FixedLenFeature([], tf.string),
                        'label'  : tf.FixedLenFeature([], tf.int64)
                    },
                    name = 'parse_op')
        img = tf.decode_raw(parse_op['img'], tf.uint8, name = 'byte_to_int8_op')
        img = tf.cast(img, tf.float32, name = 'int8_to_int32_op')
        img = tf.reshape(img, shape=[224, 224, 3], name = 'reshape_op') #Can be used for on-the-fly reshape
        label = tf.cast(parse_op['label'], tf.int32, name = 'label_cast_op')
        label = tf.one_hot(indices = label, depth = 10, axis=-1)
        return img, label

    def train_data_pipeline(self, train_data_file, PREFETCH_BUFFER, NUM_EPOCHS, BATCH_SIZE, GPU_ENABLED):
        dataset = tf.data.TFRecordDataset(filenames = train_data_file,
                                            num_parallel_reads = 8)
        dataset = dataset.apply(
                    tf.contrib.data.shuffle_and_repeat(PREFETCH_BUFFER, NUM_EPOCHS))
        dataset = dataset.apply(
                    tf.contrib.data.map_and_batch(self.tfrecord_parser, BATCH_SIZE))
        if GPU_ENABLED:
            dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
        return dataset

    def data_provider(self, args):
        file_name = self.absolute_path + "/dataset/" + self.dataset_name + "/pre-processed/" + "dataset.tfrecords"
        dataset = self.train_data_pipeline(file_name, args['PREFETCH_BUFFER'],
                    args['NUM_EPOCHS'], args['BATCH_SIZE'], args['GPU_ENABLED'])
        iterator = dataset.make_one_shot_iterator()
        img_batch, label_batch = iterator.get_next()
        return img_batch, label_batch


#d = DataSet("CIFAR-10")
#d.get_data("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
