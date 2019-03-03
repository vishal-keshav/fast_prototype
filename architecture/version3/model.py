import tensorflow as tf
import numpy as np

"""
Simple Architecture
"""

class SimpleNet:
    def __init__(self):
        self.layers = []
        self.network = {}

    """
    Builds the model
    """
    def build_model(self, img):
        #x_image = tf.reshape(img, [-1,28,28,1])
        with tf.name_scope("layer_1"):
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])
            self.network['h_conv1'] = tf.nn.relu(
                                            self.conv2d(img, W_conv1) + b_conv1)
            self.network['h_pool1'] = self.max_pool_2x2(self.network['h_conv1'])

        with tf.name_scope("layer_2"):
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            conv = self.conv2d(self.network['h_pool1'], W_conv2)
            self.network['h_conv2'] = tf.nn.relu( conv + b_conv2 )
            self.network['h_pool2'] = self.max_pool_2x2(self.network['h_conv2'])

        with tf.name_scope("layer_3"):
            W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])
            self.network['h_pool2_flat'] = tf.reshape(self.network['h_pool2'],
                                                                   [-1, 7*7*64])
            self.network['h_fc1'] = tf.nn.relu(tf.matmul(
                                   self.network['h_pool2_flat'], W_fc1) + b_fc1)

        with tf.name_scope("layer_4"):
            W_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])
            self.network['logits'] = tf.matmul(
                                           self.network['h_fc1'], W_fc2) + b_fc2
            self.network['prob'] = tf.nn.softmax(self.network['logits'])

    def weight_variable(self, shape):
        with tf.name_scope("weight"):
            initial = tf.truncated_normal(shape, stddev=0.1)
            var = tf.Variable(initial)
            variable_summaries(var)
        return var

    def bias_variable(self, shape):
        with tf.name_scope("bias"):
            initial = tf.constant(0.1, shape=shape)
            var = tf.Variable(initial)
            variable_summaries(var)
        return var

    def conv2d(self, x, W):
        with tf.name_scope("conv"):
            conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return conv

    def max_pool_2x2(self, x):
        with tf.name_scope("max_pool"):
            pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        return pool

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

"""
Model interface for creating and returning network with initializer specified
"""
def create_model(img, args):
    net = SimpleNet()
    net.build_model(img)
    return {'feature_in': img,
            'feature_logits': net.get_logits() ,
            'feature_out': net.get_feature(),
            'layer_1': net.get_network()['h_pool1'],
            'layer_2': net.get_network()['h_pool2']}
