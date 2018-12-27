import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

"""
MobileNet Architecture
"""

class MobileNet:
    def __init__(self):
        self.layers = []
        self.network = {}

    """
    Builds the model
    """
    def build_model(self, img, resolution_multiplier, width_multiplier,
                    depth_multiplier):
        out_dim = 10
        layer_1_conv = slim.convolution2d(img, round(32 * width_multiplier),
                        [3, 3], stride=2, padding='SAME', scope='conv_1')
        layer_2_dw = self.dw_separable(layer_1_conv, 64, width_multiplier,
                            depth_multiplier, sc='conv_ds_2')
        layer_3_dw = self.dw_separable(layer_2_dw, 128, width_multiplier,
                            depth_multiplier, downsample=True, sc='conv_ds_3')
        layer_4_dw = self.dw_separable(layer_3_dw, 128, width_multiplier,
                            depth_multiplier, sc='conv_ds_4')
        layer_5_dw = self.dw_separable(layer_4_dw, 256, width_multiplier,
                            depth_multiplier, downsample=True, sc='conv_ds_5')
        layer_6_dw = self.dw_separable(layer_5_dw, 256, width_multiplier,
                            depth_multiplier, sc='conv_ds_6')
        layer_7_dw = self.dw_separable(layer_6_dw, 512, width_multiplier,
                            depth_multiplier, downsample=True, sc='conv_ds_7')
        # repeatable layers can be put inside a loop
        layer_8_12_dw = layer_7_dw
        for i in range(8, 13):
            layer_8_12_dw = self.dw_separable(layer_8_12_dw, 512, width_multiplier,
                                        depth_multiplier, sc='conv_ds_'+str(i))
        layer_13_dw = self.dw_separable(layer_8_12_dw, 1024, width_multiplier,
                            depth_multiplier, downsample=True, sc='conv_ds_13')
        layer_14_dw = self.dw_separable(layer_13_dw, 1024, width_multiplier,
                            depth_multiplier, sc='conv_ds_14')
        # Pool and reduce to output dimension
        global_pool = tf.reduce_mean(layer_14_dw, [1, 2], keep_dims=True,
                                        name='global_pool')
        spatial_reduction = tf.squeeze(global_pool, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(spatial_reduction, out_dim,
                                        activation_fn=None, scope='fc_16')
        self.network['logits'] = logits
        self.network['prob'] = slim.softmax(logits, scope='Predictions')

    def dw_separable(self, input, nr_filters, width_multiplier,
                        depth_multiplier, sc, downsample=False):
        nr_filters = round(nr_filters * width_multiplier)
        if downsample:
            stride = 2
        else:
            stride = 1
        depthwise_conv = slim.separable_convolution2d(input, num_outputs=None,
                            stride=stride, depth_multiplier=depth_multiplier,
                            kernel_size=[3, 3], scope= sc +'/depthwise_conv')
        pointwise_conv = slim.convolution2d(depthwise_conv, nr_filters, kernel_size=[1, 1],
                            scope=sc +'/pointwise_conv')
        return pointwise_conv

    def get_feature(self):
        return self.network['prob']

    def get_logits(self):
        return self.network['logits']

"""
Model interface for creating and returning network with initializer specified
"""
def create_model(img, args):
    net = MobileNet()
    net.build_model(img, args['resolution_multiplier'], args['width_multiplier'], args['depth_multiplier'])
    return {'feature_in': img, 'feature_logits': net.get_logits() ,'feature_out': net.get_feature()}
