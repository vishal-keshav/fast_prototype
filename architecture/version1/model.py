import tensorflow as tf
import numpy as np
import os
import wget

"""
VGGNet Architecture
Hyper-parameter list:
* initializer: imagenet path, None
* trainable: true or false
* dropout: dropout if trainable
"""

class VGGNet:
    """
    VGG19 architecture.
    It can be trained with imageNet initialization.
    """
    def __init__(self, weights = None, dropout = 1.0, trainable = True):
        if weights is not None:
            self.data_dict = weights
        else:
            self.data_dict = None
        self.trainable = trainable
        self.dropout = dropout
        self.network = {}

    """
    Wrapper on varable re-use
    """
    def build_model(self, img, var_reuse = False):
        if var_reuse:
            with tf.variable_scope("vgg16", reuse = tf.AUTO_REUSE):
                self._build_model(img)
        else:
            self._build_model(img)

    """
    Builds the model
    """
    def _build_model(self, img):
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            input = img-mean

        with tf.name_scope('conv1_1') as scope:
            kernel = self.get_var(shape = [3, 3, 3, 64], name = "conv1_1_W")
            biases = self.get_var(shape=[64], name="conv1_1_b")
            conv1_1 = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
            conv1_1 = tf.nn.bias_add(conv1_1, biases)
            conv1_1 = tf.nn.relu(conv1_1, name=scope)
            self.network['conv1_1'] = conv1_1

        with tf.name_scope('conv1_2') as scope:
            kernel = self.get_var(shape = [3, 3, 64, 64], name = "conv1_2_W")
            biases = self.get_var(shape=[64], name="conv1_2_b")
            conv1_2 = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv1_2 = tf.nn.bias_add(conv1_2, biases)
            conv1_2 = tf.nn.relu(conv1_2, name=scope)

        with tf.name_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=scope)

        with tf.name_scope('conv2_1') as scope:
            kernel = self.get_var(shape = [3, 3, 64, 128], name = "conv2_1_W")
            biases = self.get_var(shape=[128], name="conv2_1_b")
            conv2_1 = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            conv2_1 = tf.nn.bias_add(conv2_1, biases)
            conv2_1 = tf.nn.relu(conv2_1, name=scope)

        with tf.name_scope('conv2_2') as scope:
            kernel = self.get_var(shape = [3, 3, 128, 128], name = "conv2_2_W")
            biases = self.get_var(shape=[128], name="conv2_2_b")
            conv2_2 = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv2_2 = tf.nn.bias_add(conv2_2, biases)
            conv2_2 = tf.nn.relu(conv2_2, name=scope)

        with tf.name_scope('pool2') as scope:
            pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=scope)

        with tf.name_scope('conv3_1') as scope:
            kernel = self.get_var(shape = [3, 3, 128, 256], name = "conv3_1_W")
            biases = self.get_var(shape=[256], name="conv3_1_b")
            conv3_1 = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_1 = tf.nn.bias_add(conv3_1, biases)
            conv3_1 = tf.nn.relu(conv3_1, name=scope)

        with tf.name_scope('conv3_2') as scope:
            kernel = self.get_var(shape = [3, 3, 256, 256], name = "conv3_2_W")
            biases = self.get_var(shape=[256], name="conv3_2_b")
            conv3_2 = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_2 = tf.nn.bias_add(conv3_2, biases)
            conv3_2 = tf.nn.relu(conv3_2, name=scope)

        with tf.name_scope('conv3_3') as scope:
            kernel = self.get_var(shape = [3, 3, 256, 256], name = "conv3_3_W")
            biases = self.get_var(shape=[256], name="conv3_3_b")
            conv3_3 = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            conv3_3 = tf.nn.bias_add(conv3_3, biases)
            conv3_3 = tf.nn.relu(conv3_3, name=scope)

        with tf.name_scope('pool3') as scope:
            pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=scope)

        with tf.name_scope('conv4_1') as scope:
            kernel = self.get_var(shape = [3, 3, 256, 512], name = "conv4_1_W")
            biases = self.get_var(shape=[512], name="conv4_1_b")
            conv4_1 = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_1 = tf.nn.bias_add(conv4_1, biases)
            conv4_1 = tf.nn.relu(conv4_1, name=scope)

        with tf.name_scope('conv4_2') as scope:
            kernel = self.get_var(shape = [3, 3, 512, 512], name = "conv4_2_W")
            biases = self.get_var(shape=[512], name="conv4_2_b")
            conv4_2 = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_2 = tf.nn.bias_add(conv4_2, biases)
            conv4_2 = tf.nn.relu(conv4_2, name=scope)

        with tf.name_scope('conv4_3') as scope:
            kernel = self.get_var(shape = [3, 3, 512, 512], name = "conv4_3_W")
            biases = self.get_var(shape=[512], name="conv4_3_b")
            conv4_3 = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            conv4_3 = tf.nn.bias_add(conv4_3, biases)
            conv4_3 = tf.nn.relu(conv4_3, name=scope)

        with tf.name_scope('pool4') as scope:
            pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=scope)

        with tf.name_scope('conv5_1') as scope:
            kernel = self.get_var(shape = [3, 3, 512, 512], name = "conv5_1_W")
            biases = self.get_var(shape=[512], name="conv5_1_b")
            conv5_1 = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            conv5_1 = tf.nn.bias_add(conv5_1, biases)
            conv5_1 = tf.nn.relu(conv5_1, name=scope)

        with tf.name_scope('conv5_2') as scope:
            kernel = self.get_var(shape = [3, 3, 512, 512], name = "conv5_2_W")
            biases = self.get_var(shape=[512], name="conv5_2_b")
            conv5_2 = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            conv5_2 = tf.nn.bias_add(conv5_2, biases)
            conv5_2 = tf.nn.relu(conv5_2, name=scope)

        with tf.name_scope('conv5_3') as scope:
            kernel = self.get_var(shape = [3, 3, 512, 512], name = "conv5_3_W")
            biases = self.get_var(shape=[512], name="conv5_3_b")
            conv5_3 = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            conv5_3 = tf.nn.bias_add(conv5_3, biases)
            conv5_3 = tf.nn.relu(conv5_3, name=scope)

        with tf.name_scope('pool5') as scope:
            pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=scope)

        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            kernel = self.get_var(shape = [shape, 4096], name = "fc6_W")
            biases = self.get_var(shape=[4096], name="fc6_b")
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc6 = tf.nn.bias_add(tf.matmul(pool5_flat, kernel), biases)
            fc6 = tf.nn.relu(fc6)

        with tf.name_scope('fc7') as scope:
            kernel = self.get_var(shape = [4096, 4096], name = "fc7_W")
            biases = self.get_var(shape=[4096], name="fc7_b")
            fc7 = tf.nn.bias_add(tf.matmul(fc6, kernel), biases)
            fc7 = tf.nn.relu(fc7)

        with tf.name_scope('fc8') as scope:
            kernel = self.get_var(shape = [4096, 1000], name = "fc8_W")
            biases = self.get_var(shape=[1000], name="fc8_b")
            fc8 = tf.nn.bias_add(tf.matmul(fc7, kernel), biases)
            self.network['logits'] = fc8

        with tf.name_scope('prob_out'):
            self.network['prob'] = tf.nn.softmax(fc8)

    def get_var(self, name, shape):
        if self.data_dict is not None and name in self.data_dict.keys():
            var = tf.get_variable(initializer=tf.constant(self.data_dict[name]), dtype = tf.float32, trainable = self.trainable, name = name)
        else:
            var = tf.get_variable(name = name, shape = shape, dtype = tf.float32, initializer=tf.truncated_normal_initializer(0.0, 0.001),
            trainable = self.trainable)
        variable_summaries(var)
        return var

    def get_feature(self):
        return self.network['prob']

    def get_logits(self):
        return self.network['logits']

    def get_network(self):
        return self.network

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def get_vgg_weights():
    url = "https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz"
    project_path = os.getcwd()
    vgg_weight_path = project_path + "/architecture/version1/"
    if not os.path.isfile(vgg_weight_path + "vgg16_weights.npz"):
        print("Weight file does not exist, downloading.")
        filename = wget.download(url, vgg_weight_path)
        print("Downloaded "+ filename)
    vgg_weights = np.load(vgg_weight_path + "vgg16_weights.npz")
    return vgg_weights


def create_model(img, args):
    vgg_weights = get_vgg_weights()
    #vgg_weights = None
    net = VGGNet(vgg_weights, args['dropout'], False)
    net.build_model(img, True)
    return {'feature_in': img, 'feature_logits': net.get_logits() ,'feature_out': net.get_feature(), 'layer_1': net.get_network()['conv1_1']}
