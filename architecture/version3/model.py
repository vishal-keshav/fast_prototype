# MobileNet with transfer learning cabaility.
import tensorflow as tf
import numpy as np

import os
import shutil
import wget
import tarfile

class MobileNet:
    """
    MobileNet Architecture version 1. The model has both train and validation
    architecture construction options. Weight transfer from original mobilenet
    is also possible.
    """
    def __init__(self, weights = None, train_mode = False, trainable = True,
                    nr_class = 1001):
        """
        Initialization of MobileNetv1

        Args:
            weights: Dictionary with keys as layer name and value as numpy array
            train_mode: If the network graph constructed is for training or test
            trainable: Are the variables created to construct the net may change
            nr_class: Number of class required to be output.

        Returns:
            Object of MobileNetv1

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
            print("do pre-processing in dataset module")
            input = img

        with tf.variable_scope('layer_1'):
            self.network['layer_1_out'] = self.conv(input, kernel_shape = [3,3],
                      out_channel=32, strides = [1,2,2,1], name ="layer_1/conv")

        with tf.variable_scope('layer_2'):
            self.network['layer_2_out_1'] = self.depth_conv(
                            self.network['layer_1_out'], kernel_shape = [3,3],
                                                    name = "layer_2/depth_conv")
            self.network['layer_2_out_2']= self.conv(
                            self.network['layer_2_out_1'], kernel_shape = [1,1],
                 out_channel=64, strides = [1,1,1,1],name ="layer_2/point_conv")

        with tf.variable_scope('layer_3'):
            self.network['layer_3_out_1'] = self.depth_conv(
                            self.network['layer_2_out_2'], kernel_shape = [3,3],
                                                     name ="layer_3/depth_conv")
            self.network['layer_3_out_2']= self.conv(
                            self.network['layer_3_out_1'], kernel_shape = [1,1],
                 out_channel=128, strides =[1,2,2,1],name ="layer_3/point_conv")

        with tf.variable_scope('layer_4'):
            self.network['layer_4_out_1'] = self.depth_conv(
                            self.network['layer_3_out_2'], kernel_shape = [3,3],
                                                     name ="layer_4/depth_conv")
            self.network['layer_4_out_2']= self.conv(
                            self.network['layer_4_out_1'], kernel_shape = [1,1],
                 out_channel=128, strides =[1,1,1,1],name ="layer_4/point_conv")

        with tf.variable_scope('layer_5'):
            self.network['layer_5_out_1'] = self.depth_conv(
                            self.network['layer_4_out_2'], kernel_shape = [3,3],
                                                     name ="layer_5/depth_conv")
            self.network['layer_5_out_2']= self.conv(
                            self.network['layer_5_out_1'], kernel_shape = [1,1],
                 out_channel=256, strides =[1,2,2,1],name ="layer_5/point_conv")

        with tf.variable_scope('layer_6'):
            self.network['layer_6_out_1'] = self.depth_conv(
                            self.network['layer_5_out_2'], kernel_shape = [3,3],
                                                     name ="layer_6/depth_conv")
            self.network['layer_6_out_2']= self.conv(
                            self.network['layer_6_out_1'], kernel_shape = [1,1],
                 out_channel=256, strides =[1,1,1,1],name ="layer_6/point_conv")

        with tf.variable_scope('layer_7'):
            self.network['layer_7_out_1'] = self.depth_conv(
                            self.network['layer_6_out_2'], kernel_shape = [3,3],
                                                     name ="layer_7/depth_conv")
            self.network['layer_7_out_2']= self.conv(
                        self.network['layer_7_out_1'], kernel_shape = [1,1],
                 out_channel=512, strides =[1,2,2,1],name ="layer_7/point_conv")

        for i in range(8,13):
            with tf.variable_scope('layer_'+str(i)):
                self.network['layer_' + str(i)+'_out_1'] = self.depth_conv(
                                self.network['layer_'+str(i-1)+'_out_2'],
                                kernel_shape = [3,3],
                                name = 'layer_'+str(i)+ "/depth_conv")
                self.network['layer_'+str(i)+'_out_2']= self.conv(
                                self.network['layer_'+str(i)+'_out_1'],
                                kernel_shape = [1,1],
                                out_channel=512, strides =[1,1,1,1],
                                name = 'layer_'+str(i) +"/point_conv")

        with tf.variable_scope('layer_13'):
            self.network['layer_13_out_1'] = self.depth_conv(
                            self.network['layer_12_out_2'],kernel_shape = [3,3],
                                                  name = "layer_13/depth_conv")
            self.network['layer_13_out_2']= self.conv(
                            self.network['layer_13_out_1'],kernel_shape = [1,1],
                 out_channel=1024, strides =[1,2,2,1],
                 name ="layer_13/point_conv")

        with tf.variable_scope('layer_14'):
            self.network['layer_14_out_1'] = self.depth_conv(
                            self.network['layer_13_out_2'],kernel_shape = [3,3],
                                                name = "layer_14/depth_conv")
            self.network['layer_14_out_2']= self.conv(
                            self.network['layer_14_out_1'],kernel_shape = [1,1],
                 out_channel=1024, strides =[1,1,1,1],
                 name = "layer_14/point_conv")

        with tf.variable_scope('average_pool'):
            self.network['average_pool'] = tf.nn.avg_pool(
                           self.network['layer_14_out_2'], ksize = [1, 7, 7, 1],
                           strides = [1, 1, 1, 1], padding = 'VALID')

        print("no dropout now")

        with tf.variable_scope('conv_out'):
            nr_depth = self.network['average_pool'].get_shape().as_list()[-1]
            p_kernel = self.get_var(shape = [1,1,nr_depth, self.nr_classes],
                                                       name = "conv_out/conv_W")
            p_biases = self.get_var(shape=[self.nr_classes],
                                                       name = "conv_out/conv_b")
            p_conv = tf.nn.conv2d(self.network['average_pool'], p_kernel,
                                          strides = [1,1,1,1], padding = 'SAME')
            p_conv = tf.nn.bias_add(p_conv, p_biases)
            self.network['conv_out'] = p_conv

        with tf.variable_scope('flatten'):
            nr_elems = np.prod([nr_elem
             for nr_elem in self.network['conv_out'].get_shape().as_list()[1:]])
            shape = [-1, nr_elems]
            self.network['flatten'] = tf.reshape(self.network['conv_out'],shape)

        self.network['logits'] = self.network['flatten']
        self.network['prob'] = tf.nn.softmax(self.network['logits'])
        self.network['argmax'] = tf.argmax(self.network['prob'], axis = -1,
                                                         output_type = tf.int32)

    def get_feature(self):
        return self.network['prob']

    def get_logits(self):
        return self.network['logits']

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

    def get_initializer_zeros(self, name, shape):
        if self.data_dict is not None and name in self.data_dict.keys():
            init = tf.constant_initializer(self.data_dict[name])
        else:
            init = tf.zeros_initializer()
        return init

    def get_initializer_ones(self, name, shape):
        if self.data_dict is not None and name in self.data_dict.keys():
            init = tf.constant_initializer(self.data_dict[name])
        else:
            init = tf.ones_initializer()
        return init

    def conv(self,input, kernel_shape, out_channel, strides, name):
        nr_depth = input.get_shape().as_list()[-1]
        kernel = self.get_var(shape = [kernel_shape[0], kernel_shape[1],
                                  nr_depth,out_channel], name = name + "_W")
        self.network[name + "_W"] = kernel
        conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
        beta = self.get_initializer_zeros(shape = [out_channel],
                                            name = name + "_bn/beta")
        gamma = self.get_initializer_ones(shape = [out_channel],
                                            name = name + "_bn/gamma")
        mean = self.get_initializer_zeros(shape = [out_channel],
                                            name = name + "_bn/moving_mean")
        variance = self.get_initializer_ones(shape = [out_channel],
                                            name = name + "_bn/moving_variance")
        batch = tf.layers.batch_normalization(conv,
                        epsilon = 0.0010000000474974513,
                        beta_initializer = beta,
                        gamma_initializer = gamma,
                        moving_mean_initializer = mean,
                        moving_variance_initializer = variance,
                        training=self.is_train, name =name+ '_bn')
        activation = tf.nn.relu6(batch)
        self.network[name + '_conv_no_activation'] = activation
        return activation

    def depth_conv(self,input, kernel_shape, name):
        nr_depth = input.get_shape().as_list()[-1]
        dw_kernel = self.get_var(shape=[3, 3, nr_depth, 1],name=name + '_W')
        self.network[name + "_W"] = dw_kernel
        conv_dw = tf.nn.depthwise_conv2d(input, dw_kernel, strides = [1,1,1,1],
                                                               padding = 'SAME')
        beta = self.get_initializer_zeros(shape = [nr_depth],
                                            name = name + "_bn/beta")
        gamma = self.get_initializer_ones(shape = [nr_depth],
                                            name = name + "_bn/gamma")
        mean = self.get_initializer_zeros(shape = [nr_depth],
                                            name = name + "_bn/moving_mean")
        variance = self.get_initializer_ones(shape = [nr_depth],
                                            name = name + "_bn/moving_variance")
        batch=tf.layers.batch_normalization(conv_dw,
                        epsilon = 0.0010000000474974513,
                        beta_initializer = beta,
                        gamma_initializer = gamma,
                        moving_mean_initializer = mean,
                        moving_variance_initializer = variance,
                        training=self.is_train,
                        name =name+ '_bn')
        activation = tf.nn.relu6(batch)
        return activation

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

#---------------------------------Below are the utilities to get pre-trained
#---------------------------------weights for mobilenetv1 from official site
def create_var_map_dict():
    """
    Doing the transfer learning
    The data variable used by this model are as follow:
    For i == 1, layer_<i>/conv_W:0,
                (MobilenetV1/Conv2d_<i-1>/weights:0)
                layer_<i>/conv_bn/gamma:0,
                (MobilenetV1/Conv2d_<i-1>/BatchNorm/gamma:0)
                layer_<i>/conv_bn/beta:0
                (MobilenetV1/Conv2d_<i-1>/BatchNorm/beta:0)
                layer_<i>/conv_bn/moving_mean:0
                (MobilenetV1/Conv2d_<i-1>/BatchNorm/moving_mean:0)
                layer_<i>/conv_bn/moving_variance:0
                (MobilenetV1/Conv2d_<i-1>/BatchNorm/moving_variance:0)
    for i == [2, 14], layer_<i>/depth_conv_W:0,
                      (MobilenetV1/Conv2d_<i-1>_depthwise/depthwise_weights:0)
                      layer_<i>/depth_conv_bn/gamma:0,
                      (MobilenetV1/Conv2d_<i-1>_depthwise/BatchNorm/gamma:0)
                      layer_<i>/depth_conv_bn/beta:0,
                      (MobilenetV1/Conv2d_<i-1>_depthwise/BatchNorm/beta:0)
                      layer_<i>/depth_conv_bn/moving_mean:0,
                      (MobilenetV1/Conv2d_<i-1>_depthwise/BatchNorm/moving_mean:0)
                      layer_<i>/depth_conv_bn/moving_variance:0,
                      (MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance:0)
                      layer_<i>/point_conv_W:0,
                      (MobilenetV1/Conv2d_<i-1>_pointwise/weights:0)
                      layer_<i>/point_conv_bn/gamma:0,
                      (MobilenetV1/Conv2d_<i-1>_pointwise/BatchNorm/gamma:0)
                      layer_<i>/point_conv_bn/beta:0,
                      (MobilenetV1/Conv2d_<i-1>_pointwise/BatchNorm/beta:0)
                      layer_<i>/point_conv_bn/moving_mean:0,
                      (MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean:0)
                      layer_<i>/point_conv_bn/moving_variance:0,
                      (MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance:0)
    for last layer "conv_out", conv_out/conv_W:0,
                               (MobilenetV1/Logits/Conv2d_1c_1x1/weights:0)
                               conv_out/conv_b:0
                               (MobilenetV1/Logits/Conv2d_1c_1x1/biases:0)
    TOTAL DIMS OF TRAINABLE VARIABLES 4233001
    """
    var_dict = {}
    var_dict['MobilenetV1/Conv2d_0/weights:0'] = 'layer_1/conv_W'
    var_dict['MobilenetV1/Conv2d_0/BatchNorm/gamma:0']='layer_1/conv_bn/gamma'
    var_dict['MobilenetV1/Conv2d_0/BatchNorm/beta:0']='layer_1/conv_bn/beta'
    var_dict['MobilenetV1/Conv2d_0/BatchNorm/moving_mean:0'] = 'layer_1/conv_bn/moving_mean'
    var_dict['MobilenetV1/Conv2d_0/BatchNorm/moving_variance:0'] = 'layer_1/conv_bn/moving_variance'
    for i in range(2,15):
        l1 = 'layer_' + str(i)
        l2 = 'MobilenetV1/Conv2d_'+ str(i-1)
        var_dict[l2+ '_depthwise/depthwise_weights:0'] = l1+ '/depth_conv_W'
        var_dict[l2+ '_depthwise/BatchNorm/gamma:0']=l1+'/depth_conv_bn/gamma'
        var_dict[l2+'_depthwise/BatchNorm/beta:0']=l1 + '/depth_conv_bn/beta'
        var_dict[l2+'_depthwise/BatchNorm/moving_mean:0']=l1 + '/depth_conv_bn/moving_mean'
        var_dict[l2+'_depthwise/BatchNorm/moving_variance:0']=l1 + '/depth_conv_bn/moving_variance'
        var_dict[l2+ '_pointwise/weights:0'] = l1+ '/point_conv_W'
        var_dict[l2+ '_pointwise/BatchNorm/gamma:0']=l1+'/point_conv_bn/gamma'
        var_dict[l2+'_pointwise/BatchNorm/beta:0']=l1 + '/point_conv_bn/beta'
        var_dict[l2+ '_pointwise/BatchNorm/moving_mean:0']=l1+'/point_conv_bn/moving_mean'
        var_dict[l2+'_pointwise/BatchNorm/moving_variance:0']=l1 + '/point_conv_bn/moving_variance'
    var_dict['MobilenetV1/Logits/Conv2d_1c_1x1/weights:0'] = 'conv_out/conv_W'
    var_dict['MobilenetV1/Logits/Conv2d_1c_1x1/biases:0'] = 'conv_out/conv_b'
    return var_dict

def create_variable_dict_from_session_graph(sess, var_name_map):
    var_dict = {}
    #TODO: check if var_name_map contains the variable.name key
    for variable in tf.moving_average_variables():
        var_dict[var_name_map[variable.name]] = variable.eval(sess)
    return var_dict

def get_mobilenet_weights_from_checkpoint():
    url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/' + \
            'mobilenet_v1_1.0_224.tgz'
    project_path = os.getcwd()
    mobilenet_weight_path = project_path + "/architecture/version3/"
    print("Downloading mobilenet checkpoints")
    filename = wget.download(url, mobilenet_weight_path)
    print("Downloaded " + filename)
    tar = tarfile.open(filename)
    tar.extractall(path = mobilenet_weight_path + 'mobilenet_v1_1.0_224')
    tar.close()
    dir_name = mobilenet_weight_path + 'mobilenet_v1_1.0_224'
    ckpt_name = dir_name + '/mobilenet_v1_1.0_224.ckpt'
    g = tf.Graph()
    with g.as_default() as g_temp:
        with tf.Session(graph = g_temp) as sess:
            if tf.train.checkpoint_exists(ckpt_name):
                print("checkpoint found, restoring graph")
                new_saver = tf.train.import_meta_graph(ckpt_name + '.meta',
                                                             clear_devices=True)
                new_saver.restore(sess, dir_name + '/mobilenet_v1_1.0_224.ckpt')
                var_map = create_var_map_dict()
                weight_dict=create_variable_dict_from_session_graph(sess,var_map)
            else:
                print("Not able to restore checkpoint")
                weight_dict = None
    print("Creating npz for mobilenet weights")
    np.savez(mobilenet_weight_path + 'mobilenet_weights.npz', **weight_dict)
    print("Cleanup checkpoints")
    os.unlink(mobilenet_weight_path + 'mobilenet_v1_1.0_224.tgz')
    shutil.rmtree(mobilenet_weight_path + 'mobilenet_v1_1.0_224')

def get_mobilenet_weights():
    project_path = os.getcwd()
    mobilenet_weight_path = project_path + "/architecture/version3/"
    if not os.path.isfile(mobilenet_weight_path + "mobilenet_weights.npz"):
        get_mobilenet_weights_from_checkpoint()
    weight_dict = np.load(mobilenet_weight_path + "mobilenet_weights.npz")
    return weight_dict

def create_model(img, is_train = True, pretrained = False):
    if pretrained:
        mobilenet_weights = get_mobilenet_weights()
    else:
        mobilenet_weights = None
    net = MobileNet(mobilenet_weights, is_train)
    net.build_model(img)
    return {'feature_in': img,
            'feature_logits': net.get_logits(),
            'feature_out': net.get_feature(),
            'network': net.get_network()}
