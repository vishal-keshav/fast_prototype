# SimpleNet architecture construction
import tensorflow as tf
import numpy as np

class SimpleNet:
    """
    Simple Architecture, as detailed below:
    Layer1: Conv (5,5) with 32 filters.
    Layer2: Conv (5,5) with 64 filters
    Layer3: Dense layer with 1024 outputs
    Layer4: Dense layer with nr_class outputs
    """
    def __init__(self, weights = None, train_mode = False, trainable = True,
                    nr_class = 10):
        """
        Initialization of Simplenet

        Args:
            weights: Dictionary with keys as layer name and value as numpy array
            train_mode: If the network graph constructed is for training or test
            trainable: Are the variables created to construct the net may change
            nr_class: Number of class required to be output.

        Returns:
            Object of SimpleNet

        Raises:
            KeyError: None
        """
        self.network = {}
        self.is_train = train_mode
        self.nr_classes = nr_class
        if weights is not None:
            self.data_dict = weights
        else:
            self.data_dict = None
        self.trainable = trainable

    def build_model(self, img):
        """
        The parent function to build the network. The auto reuse of variables is
        enabled by default.

        Args:
            img: The image tensor, should have some dimention(except first axis)
                 Tensors can be Placeholders, dataset tensors, etc.

        Returns:
            Nothing is returned, tf graph will be constructed.
        """
        with tf.variable_scope("", reuse = tf.AUTO_REUSE):
            self._build_model(img)

    def _build_model(self, img):
        input_shape = img.get_shape().as_list()
        with tf.variable_scope("layer_1"):
            W_conv1 = self.weight_variable([5, 5, input_shape[3], 32])
            b_conv1 = self.bias_variable([32])
            self.network['W_conv1'] = W_conv1
            self.network['b_conv1'] = b_conv1
            self.network['h_conv1'] = self.conv2d(img, W_conv1) + b_conv1
            self.network['conv1_ac'] = tf.nn.relu(self.network['h_conv1'])
            self.network['h_pool1'] =self.max_pool_2x2(self.network['conv1_ac'])

        with tf.variable_scope("layer_2"):
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            conv = self.conv2d(self.network['h_pool1'], W_conv2)
            self.network['W_conv2'] = W_conv2
            self.network['b_conv2'] = b_conv2
            self.network['h_conv2'] = conv + b_conv2
            self.network['conv2_ac'] = tf.nn.relu( self.network['h_conv2'] )
            self.network['h_pool2'] =self.max_pool_2x2(self.network['conv2_ac'])

        with tf.variable_scope("layer_3"):
            pool_shape = self.network['h_pool2'].get_shape().as_list()
            nr_elem = pool_shape[1]*pool_shape[2]*pool_shape[3]
            W_fc1 = self.weight_variable([nr_elem, 1024])
            b_fc1 = self.bias_variable([1024])
            self.network['W_fc1'] = W_fc1
            self.network['b_fc1'] = b_fc1
            self.network['h_pool2_flat'] = tf.reshape(self.network['h_pool2'],
                                                                   [-1,nr_elem])
            self.network['h_fc1'] = tf.nn.relu(tf.matmul(
                                   self.network['h_pool2_flat'], W_fc1) + b_fc1)

        with tf.variable_scope("layer_4"):
            W_fc2 = self.weight_variable([1024, self.nr_classes])
            b_fc2 = self.bias_variable([self.nr_classes])
            self.network['W_fc2'] = W_fc2
            self.network['b_fc2'] = b_fc2
            self.network['logits'] = tf.matmul(
                                           self.network['h_fc1'], W_fc2) + b_fc2
            self.network['prob'] = tf.nn.softmax(self.network['logits'])

    def weight_variable(self, shape):
        with tf.variable_scope("weight") as scope:
            initial = tf.truncated_normal(shape, stddev=0.1)
            print(scope)
            var = tf.get_variable(initializer = initial, dtype = tf.float32,
                        trainable = self.trainable, name = "weight")
            variable_summaries(var)
        return var

    def bias_variable(self, shape):
        with tf.variable_scope("bias") as scope:
            initial = tf.constant(0.1, shape=shape)
            var = tf.get_variable(initializer = initial, dtype = tf.float32,
                        trainable = self.trainable, name = "bias")
            variable_summaries(var)
        return var

    def conv2d(self, x, W):
        with tf.variable_scope("conv"):
            conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return conv

    def max_pool_2x2(self, x):
        with tf.variable_scope("max_pool"):
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
    with tf.variable_scope('summaries'):
        mean = tf.reduce_mean(var)
    with tf.variable_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

"""
Model interface for creating and returning network with initializer specified
SimpleNet can be easily initialized and APIs can be used to construct the
network, but we use a simple functional interface to get and build simplenet
"""
def create_model(img, is_train = True, pretrained = False):
    net = SimpleNet(train_mode = is_train)
    net.build_model(img)
    return {'feature_in': img,
            'feature_logits': net.get_logits() ,
            'feature_out': net.get_feature(),
            'network': net.get_network()}

if __name__ == "__main__":
    img = tf.random.uniform(shape = (28,28,3,3))
    arch_dict = create_model(img)
    print(arch_dict)
