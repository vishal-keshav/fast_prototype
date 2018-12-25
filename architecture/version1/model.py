import tensorflow as tf
import numpy as np

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
    def __init__(self, initializer = None, trainable = True):
        if initializer is not None:
            self.data_dict = np.load(initializer)[()]
        else:
            self.data_dict = None
        self.trainable = trainable
        self.layers = []
        self.network = {}
        self.var_dict = {}

    """
    Builds the model
    """
    def build_model(self, img, dropout):
        train_mode = None
        self.dropout = dropout
        blue, green, red = tf.split(axis = 3, num_or_size_splits = 3, value=img)
        print(blue.get_shape().as_list()[1:])
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis = 3, values = [blue, green ,red])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        self.network['conv1_1'] = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.network['conv1_2'] = self.conv_layer(self.network['conv1_1'], 64, 64, "conv1_2")
        self.network['pool1'] = self.max_pool(self.network['conv1_2'], 'pool1')

        self.network['conv2_1'] = self.conv_layer(self.network['pool1'], 64, 128, "conv2_1")
        self.network['conv2_2'] = self.conv_layer(self.network['conv2_1'], 128, 128, "conv2_2")
        self.network['pool2'] = self.max_pool(self.network['conv2_2'], 'pool2')

        self.network['conv3_1'] = self.conv_layer(self.network['pool2'], 128, 256, "conv3_1")
        self.network['conv3_2'] = self.conv_layer(self.network['conv3_1'], 256, 256, "conv3_2")
        self.network['conv3_3'] = self.conv_layer(self.network['conv3_2'], 256, 256, "conv3_3")
        self.network['conv3_4'] = self.conv_layer(self.network['conv3_3'], 256, 256, "conv3_4")
        self.network['pool3'] = self.max_pool(self.network['conv3_4'], 'pool3')

        self.network['conv4_1'] = self.conv_layer(self.network['pool3'], 256, 512, "conv4_1")
        self.network['conv4_2'] = self.conv_layer(self.network['conv4_1'], 512, 512, "conv4_2")
        self.network['conv4_3'] = self.conv_layer(self.network['conv4_2'], 512, 512, "conv4_3")
        self.network['conv4_4'] = self.conv_layer(self.network['conv4_3'], 512, 512, "conv4_4")
        self.network['pool4'] = self.max_pool(self.network['conv4_4'], 'pool4')

        self.network['conv5_1'] = self.conv_layer(self.network['pool4'], 512, 512, "conv5_1")
        self.network['conv5_2'] = self.conv_layer(self.network['conv5_1'], 512, 512, "conv5_2")
        self.network['conv5_3'] = self.conv_layer(self.network['conv5_2'], 512, 512, "conv5_3")
        self.network['conv5_4'] = self.conv_layer(self.network['conv5_3'], 512, 512, "conv5_4")
        self.network['pool5'] = self.max_pool(self.network['conv5_4'], 'pool5')

        self.network['fc6'] = self.fc_layer(self.network['pool5'], 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.network['relu6'] = tf.nn.relu(self.network['fc6'])
        if train_mode is not None:
            self.network['relu6'] = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.network['relu6'])
        elif self.trainable:
            self.network['relu6'] = tf.nn.dropout(self.network['relu6'], self.dropout)

        self.network['fc7'] = self.fc_layer(self.network['relu6'], 4096, 4096, "fc7")
        self.network['relu7'] = tf.nn.relu(self.network['fc7'])
        if train_mode is not None:
            self.network['relu7'] = tf.cond(train_mode, lambda: tf.nn.dropout(self.network['relu7'], self.dropout), lambda: self.network['relu7'])
        elif self.trainable:
            self.network['relu7'] = tf.nn.dropout(self.network['relu7'], self.dropout)

        self.network['fc8'] = self.fc_layer(self.network['relu7'], 4096, 1000, "fc8")
        self.network['fc9'] = self.fc_layer(self.network['fc8'], 1000, 10, "fc8")

        self.network['prob'] = tf.nn.softmax(self.network['fc9'], name="prob")
        # Empty the data dictionary, model has been initialized
        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            self.layers.append(conv)
            self.layers.append(relu)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            self.layers.append(fc)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        weights = self.get_var(name + "_weights", [filter_size, filter_size, in_channels, out_channels])
        biases = self.get_var(name + "_biases", [out_channels])

        return weights, biases

    def get_fc_var(self, in_size, out_size, name):
        weights = self.get_var(name + "_weights", [in_size, out_size])
        biases = self.get_var(name + "_biases", [out_size])

        return weights, biases

    def get_var(self, name, shape):
        if self.data_dict is not None and name in self.data_dict:
            placeholder = tf.placeholder(tf.float32, shape)
            var = tf.Variable(placeholder, name=name)
            self.var_dict[placeholder] = self.data_dict[name]
        else:
            var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(0.0, 0.001))

        return var

    def get_feature(self):
        return self.network['prob']

"""
Model interface for creating and returning network with initializer specified
"""
def create_model(img, args):
    net = VGGNet()
    net.build_model(img, args['dropout'])
    return {'feature_in': img, 'feature_logits': self.network['fc9'] ,'feature_out': net.get_feature()}
