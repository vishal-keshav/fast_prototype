# Unet architecture for image reconstruction
import tensorflow as tf
import numpy as np

class UNet:
    """
    UNet architecture that supports FCN. The model has both train and validation
    architecture construction options. Weight transfer from original LTSID
    is also possible.
    """
    def __init__(self, weights = None, train_mode = False, trainable = True,
                    nr_class = 0):
        """
        Initialization of UNet

        Args:
            weights: Dictionary with keys as layer name and value as numpy array
            train_mode: If the network graph constructed is for training or test
            trainable: Are the variables created to construct the net may change
            nr_class: Number of class required to be output.

        Returns:
            Object of Unet

        Raises:
            KeyError: None
        """
        self.layers = []
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
        Wrapper with variable resue on _build_model that does the heavy lifting.
        """
        with tf.variable_scope("", reuse = tf.AUTO_REUSE):
            self._build_model(img)

    def _build_model(self, img):
        with tf.name_scope('preprocess') as scope:
            input = self.pack_raw_tf(img)
            input = tf.expand_dims(input, axis = 0)*100
            input = tf.math.minimum(input, 1.0)
            self.network['packed'] = input
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_1'):
            self.network['layer_1_out'] = self.conv(input, kernel_shape = [3,3],
                      out_channel=32, strides = [1,1,1,1], name ="layer_1/conv")
            self.network['layer_1_out'] =self.lrelu(self.network['layer_1_out'])

        with tf.variable_scope('layer_2'):
            self.network['layer_2_out'] = self.conv(
                              self.network['layer_1_out'], kernel_shape = [3,3],
                     out_channel=32, strides = [1,1,1,1], name = "layer_2/conv")
            self.network['layer_2_out'] =self.lrelu(self.network['layer_2_out'])
            self.network['layer_2_pool'] = self.max_pool(
              self.network['layer_2_out'], kernel_shape = [2,2], padding='SAME')
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_3'):
            self.network['layer_3_out'] = self.conv(
                            self.network['layer_2_pool'], kernel_shape = [3,3],
                      out_channel=64, strides = [1,1,1,1], name ="layer_3/conv")
            self.network['layer_3_out'] =self.lrelu(self.network['layer_3_out'])

        with tf.variable_scope('layer_4'):
            self.network['layer_4_out'] = self.conv(
                              self.network['layer_3_out'], kernel_shape = [3,3],
                     out_channel=64, strides = [1,1,1,1], name = "layer_4/conv")
            self.network['layer_4_out'] =self.lrelu(self.network['layer_4_out'])
            self.network['layer_4_pool'] = self.max_pool(
              self.network['layer_4_out'], kernel_shape = [2,2], padding='SAME')
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_5'):
            self.network['layer_5_out'] = self.conv(
                             self.network['layer_4_pool'], kernel_shape = [3,3],
                      out_channel=128, strides = [1,1,1,1], name ="layer_5/conv")
            self.network['layer_5_out'] =self.lrelu(self.network['layer_5_out'])

        with tf.variable_scope('layer_6'):
            self.network['layer_6_out'] = self.conv(
                              self.network['layer_5_out'], kernel_shape = [3,3],
                     out_channel=128, strides = [1,1,1,1], name ="layer_6/conv")
            self.network['layer_6_out'] =self.lrelu(self.network['layer_6_out'])
            self.network['layer_6_pool'] = self.max_pool(
              self.network['layer_6_out'], kernel_shape = [2,2], padding='SAME')
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_7'):
            self.network['layer_7_out'] = self.conv(
                             self.network['layer_6_pool'], kernel_shape = [3,3],
                      out_channel=256, strides = [1,1,1,1], name ="layer_7/conv")
            self.network['layer_7_out'] =self.lrelu(self.network['layer_7_out'])

        with tf.variable_scope('layer_8'):
            self.network['layer_8_out'] = self.conv(
                              self.network['layer_7_out'], kernel_shape = [3,3],
                     out_channel=256, strides = [1,1,1,1], name ="layer_8/conv")
            self.network['layer_8_out'] =self.lrelu(self.network['layer_8_out'])
            self.network['layer_8_pool'] = self.max_pool(
              self.network['layer_8_out'], kernel_shape = [2,2], padding='SAME')
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_9'):
            self.network['layer_9_out'] = self.conv(
                             self.network['layer_8_pool'], kernel_shape = [3,3],
                      out_channel=512, strides = [1,1,1,1], name ="layer_9/conv")
            self.network['layer_9_out'] =self.lrelu(self.network['layer_9_out'])

        with tf.variable_scope('layer_10'):
            self.network['layer_10_out'] = self.conv(
                              self.network['layer_9_out'], kernel_shape = [3,3],
                     out_channel=512, strides = [1,1,1,1],name ="layer_10/conv")
            self.network['layer_10_out']=self.lrelu(self.network['layer_10_out'])
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_11'):
            self.network['layer_11_out'] = self.upsample_and_concat(
                                           self.network['layer_10_out'],
                                           self.network['layer_8_out'],256, 512)

        with tf.variable_scope('layer_12'):
            self.network['layer_12_out']= self.conv(
                            self.network['layer_11_out'], kernel_shape = [3,3],
                     out_channel=256, strides =[1,1,1,1], name ="layer_12/conv")
            self.network['layer_12_out']=self.lrelu(self.network['layer_12_out'])

        with tf.variable_scope('layer_13'):
            self.network['layer_13_out']= self.conv(
                            self.network['layer_12_out'], kernel_shape = [3,3],
                     out_channel=256, strides =[1,1,1,1], name ="layer_13/conv")
            self.network['layer_13_out']=self.lrelu(self.network['layer_13_out'])
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_14'):
            self.network['layer_14_out'] = self.upsample_and_concat(
                                           self.network['layer_13_out'],
                                           self.network['layer_6_out'],128, 256)

        with tf.variable_scope('layer_15'):
            self.network['layer_15_out']= self.conv(
                            self.network['layer_14_out'], kernel_shape = [3,3],
                     out_channel=128, strides =[1,1,1,1], name ="layer_15/conv")
            self.network['layer_15_out']=self.lrelu(self.network['layer_15_out'])

        with tf.variable_scope('layer_16'):
            self.network['layer_16_out']= self.conv(
                            self.network['layer_15_out'], kernel_shape = [3,3],
                     out_channel=128, strides =[1,1,1,1], name ="layer_16/conv")
            self.network['layer_16_out']=self.lrelu(self.network['layer_16_out'])
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_17'):
            self.network['layer_17_out'] = self.upsample_and_concat(
                                           self.network['layer_16_out'],
                                           self.network['layer_4_out'],64, 128)

        with tf.variable_scope('layer_18'):
            self.network['layer_18_out']= self.conv(
                            self.network['layer_17_out'], kernel_shape = [3,3],
                     out_channel=64, strides =[1,1,1,1], name ="layer_18/conv")
            self.network['layer_18_out']=self.lrelu(self.network['layer_18_out'])

        with tf.variable_scope('layer_19'):
            self.network['layer_19_out']= self.conv(
                            self.network['layer_18_out'], kernel_shape = [3,3],
                     out_channel=64, strides =[1,1,1,1], name ="layer_19/conv")
            self.network['layer_19_out']=self.lrelu(self.network['layer_19_out'])
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_20'):
            self.network['layer_20_out'] = self.upsample_and_concat(
                                           self.network['layer_19_out'],
                                           self.network['layer_2_out'],32, 64)

        with tf.variable_scope('layer_21'):
            self.network['layer_21_out']= self.conv(
                            self.network['layer_20_out'], kernel_shape = [3,3],
                     out_channel=32, strides =[1,1,1,1], name ="layer_21/conv")
            self.network['layer_21_out']=self.lrelu(self.network['layer_21_out'])

        with tf.variable_scope('layer_22'):
            self.network['layer_22_out']= self.conv(
                            self.network['layer_21_out'], kernel_shape = [3,3],
                     out_channel=32, strides =[1,1,1,1], name ="layer_22/conv")
            self.network['layer_22_out']=self.lrelu(self.network['layer_22_out'])
#------------------------------------------------------------------------------#
        with tf.variable_scope('layer_23'):
            self.network['layer_23_out'] = self.conv(
                            self.network['layer_22_out'], kernel_shape = [1,1],
                      out_channel=12, strides =[1,1,1,1], name ="layer_23/conv")

        with tf.variable_scope('depth_to_space'):
            self.network['out'] = tf.depth_to_space(
                                                self.network['layer_23_out'], 2)
#------------------------------------------------------------------------------#
    def get_feature(self):
        return self.network['out']

    def get_logits(self):
        return self.network['out']

    def get_network(self):
        return self.network

    def get_var(self, name, shape):
        if self.data_dict is not None and name in self.data_dict.keys():
            var = tf.get_variable(initializer=tf.constant(self.data_dict[name]),
                    dtype = tf.float32, trainable = self.trainable, name = name)
        else:
            var = tf.get_variable(name = name, shape = shape, dtype =tf.float32,
                        #initializer=tf.truncated_normal_initializer(0.0, 0.001),
                        initializer = tf.contrib.layers.xavier_initializer(),
                        trainable = self.trainable)
        variable_summaries(var)
        return var

    def conv(self,input, kernel_shape, out_channel, strides, name):
        nr_depth = input.get_shape().as_list()[-1]
        kernel = self.get_var(shape = [kernel_shape[0], kernel_shape[1],
                                  nr_depth,out_channel], name = name + "_W")
        self.network[name + "_W"] = kernel
        conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
        return conv

    def upsample_and_concat(self, tensor1, tensor2, out_channels, in_channels):
        deconv_filter = tf.Variable(tf.truncated_normal(
                                [2, 2, out_channels, in_channels], stddev=0.02))
        deconv = tf.nn.conv2d_transpose(tensor1, deconv_filter,
                                        tf.shape(tensor2), strides=[1, 2, 2, 1])

        deconv_output = tf.concat([deconv, tensor2], 3)
        deconv_output.set_shape([None, None, None, out_channels * 2])
        return deconv_output

    def lrelu(self, x):
        return tf.maximum(x * 0.2, x)

    def max_pool(self, tensor, kernel_shape, padding):
        ksize = [1]+kernel_shape+[1]
        strides = ksize
        return tf.nn.max_pool(tensor, ksize, strides, padding)

    def pack_raw_tf(self, im):
        im = tf.math.maximum(im - 0, 0)/(1024 - 0)
        im = tf.expand_dims(im, axis=2)
        img_shape = im.get_shape().as_list()
        H = img_shape[0]
        W = img_shape[1]
        out = tf.concat((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out

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

def create_model(img, is_train = True, pretrained = False):
    unet_weights = None
    net = UNet(unet_weights, is_train)
    net.build_model(img)
    return {'feature_in': img,
            'feature_logits': net.get_logits(),
            'feature_out': net.get_feature(),
            'network': net.get_network()}
