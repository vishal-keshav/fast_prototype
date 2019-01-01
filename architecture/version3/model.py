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
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        self.network['h_conv1'] = tf.nn.relu(self.conv2d(img, W_conv1) + b_conv1)
        self.network['h_pool1'] = self.max_pool_2x2(self.network['h_conv1'])

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        conv = self.conv2d(self.network['h_pool1'], W_conv2)
        self.network['h_conv2'] = tf.nn.relu( conv + b_conv2 )
        self.network['h_pool2'] = self.max_pool_2x2(self.network['h_conv2'])

        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        self.network['h_pool2_flat'] = tf.reshape(self.network['h_pool2'], [-1, 7*7*64])
        self.network['h_fc1'] = tf.nn.relu(tf.matmul(self.network['h_pool2_flat'], W_fc1) + b_fc1)

        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        self.network['logits'] = tf.matmul(self.network['h_fc1'], W_fc2) + b_fc2
        self.network['prob'] = tf.nn.softmax(self.network['logits'])

    def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(self, shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    def get_feature(self):
        return self.network['prob']

    def get_logits(self):
        return self.network['logits']

"""
Model interface for creating and returning network with initializer specified
"""
def create_model(img, args):
    net = SimpleNet()
    net.build_model(img)
    return {'feature_in': img, 'feature_logits': net.get_logits() ,'feature_out': net.get_feature()}
