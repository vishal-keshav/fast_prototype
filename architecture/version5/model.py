# YOLOv3 with transfer learning cabaility.
import tensorflow as tf
import numpy as np

import os
import shutil
import wget
import tarfile

class YOLOv3:
    """
    YOLO Architecture version 3. The model has both train and validation
    architecture construction options. Weight transfer from original YOLOv3
    is also possible.
    """
    def __init__(self, weights = None, train_mode = False, trainable = True,
                    nr_class = 80):
        """
        Initialization of YOLOv3
        Args:
            weights: Dictionary with keys as layer name and value as numpy array
            train_mode: If the network graph constructed is for training or test
            trainable: Are the variables created to construct the net may change
            nr_class: Number of class required to be output.
        Returns:
            Object of YOLOv3
        Raises:
            KeyError: None
        """
        self.layers = []
        self.network = {}
        self.is_train = train_mode
        self.nr_classes = nr_class
        self.img_height = 416
        self.img_width = 416
        self.nr_achors_per_detector = 3
        self.anchor_scale1 = [(10, 13), (16, 30), (33, 23)]
        self.anchor_scale2 = [(30, 61), (62, 45), (59, 119)]
        self.anchor_scale3 = [(116, 90), (156, 198), (373, 326)]
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
            # Override the default spatial dimentions of the image
            input_shape = input.get_shape().as_list()
            self.img_height = input_shape[1]
            self.img_width = input_shape[2]

        with tf.variable_scope('layer_1'):
            self.network['layer_1_out'] = self.conv(input, kernel_shape = [3,3],
                      out_channel=32, strides = [1,1,1,1], name ="layer_1/conv")

        with tf.variable_scope('layer_2'):
            self.network['layer_2_out'] = self.conv(self.network['layer_1_out'],
                                kernel_shape = [3,3],out_channel=64,
                                strides = [1,2,2,1], name ="layer_2/conv")

        with tf.variable_scope('layer_3'):
            self.network['layer_3_out'] = self.residual_block(
                                self.network['layer_2_out'], out_channel = 32,
                                strides = [1,1,1,1], name = "layer_3")

        with tf.variable_scope('layer_4'):
            self.network['layer_4_out'] = self.conv(self.network['layer_3_out'],
                            out_channel = 128, strides = [1,2,2,1],
                            kernel_shape = [3,3], name ="layer_4/conv")

        for index, layer in enumerate(range(5,7)):
            with tf.variable_scope('layer_'+str(layer)+'_'+str(index+1)):
                self.network['layer_'+str(layer)+'_out'] = self.residual_block(
                    self.network['layer_'+str(layer-1)+'_out'],out_channel = 64,
                    strides = [1,1,1,1], name = "layer_" + str(layer))

        with tf.variable_scope('layer_7'):
            self.network['layer_7_out'] = self.conv(self.network['layer_6_out'],
                                    kernel_shape = [3,3], out_channel = 256,
                                    strides = [1,2,2,1], name ="layer_7/conv")

        for index, layer in enumerate(range(8,16)):
            with tf.variable_scope('layer_'+str(layer)+'_'+str(index+1)):
                self.network['layer_'+str(layer)+'_out'] = self.residual_block(
                    self.network['layer_'+str(layer-1)+'_out'],out_channel =128,
                    strides = [1,1,1,1], name = "layer_" + str(layer))

        self.network['scale_1_out'] = self.network['layer_15_out']

        with tf.variable_scope('layer_16'):
            self.network['layer_16_out']=self.conv(self.network['layer_15_out'],
                                    kernel_shape = [3,3], out_channel = 512,
                                    strides = [1,2,2,1], name ="layer_16/conv")

        for index, layer in enumerate(range(17,25)):
            with tf.variable_scope('layer_'+str(layer)+'_'+str(index+1)):
                self.network['layer_'+str(layer)+'_out'] = self.residual_block(
                    self.network['layer_'+str(layer-1)+'_out'],out_channel =256,
                    strides = [1,1,1,1], name = "layer_" + str(layer))

        self.network['scale_2_out'] = self.network['layer_24_out']

        with tf.variable_scope('layer_25'):
            self.network['layer_25_out']=self.conv(self.network['layer_24_out'],
                                    kernel_shape = [3,3], out_channel = 1024,
                                    strides = [1,2,2,1], name = "layer_25/conv")

        for index, layer in enumerate(range(26,30)):
            with tf.variable_scope('layer_'+str(layer)+'_'+str(index+1)):
                self.network['layer_'+str(layer)+'_out'] = self.residual_block(
                    self.network['layer_'+str(layer-1)+'_out'],out_channel=512,
                    strides = [1,1,1,1], name = "layer_" + str(layer))
        with tf.variable_scope('layer_convblock_3'):
            self.network['scale_3_out'] = self.conv_block(
                            self.network['layer_29_out'], out_channel = 512,
                            name = "convblock_3")
#-------------------------Contruction of detectors------------------------------
        with tf.variable_scope('detection_layer_scale_3'):
            self.network['detection_scale_3'] = self.detector_block(
                            self.network['scale_3_out'], self.anchor_scale3,
                            name = "detector_scale_3")

        with tf.variable_scope('detection_layer_scale_2'):
            self.network['scale2_conv1_out'] = self.conv(
                self.network['convblock_3_intermediate'], kernel_shape = [1,1],
                out_channel = 256, strides = [1,1,1,1], name = "scale2/conv1")
            upsample_size = self.network['scale_2_out'].get_shape().as_list()
            self.network['scale2_up']  = self.upsample(
                                self.network['scale2_conv1_out'], upsample_size)
            self.network['multiscale2_concat'] = tf.concat(
                [self.network['scale2_up'],self.network['scale_2_out']],axis=3)
            self.network['scale_2_out_final'] = self.conv_block(
                        self.network['multiscale2_concat'], out_channel = 256,
                        name = "convblock_2")
            self.network['detection_scale_2'] = self.detector_block(
                        self.network['scale_2_out_final'], self.anchor_scale2,
                        name = "detector_scale_2")

        with tf.variable_scope('detection_layer_scale_1'):
            self.network['scale1_conv1_out'] = self.conv(
                self.network['convblock_2_intermediate'], kernel_shape = [1,1],
                out_channel = 128, strides = [1,1,1,1], name = "scale1/conv1")
            upsample_size = self.network['scale_1_out'].get_shape().as_list()
            self.network['scale1_up'] = self.upsample(
                                self.network['scale1_conv1_out'], upsample_size)
            self.network['multiscale1_concat'] = tf.concat(
                [self.network['scale1_up'],self.network['scale_1_out']],axis=3)
            self.network['scale_1_out_final'] = self.conv_block(
                        self.network['multiscale1_concat'], out_channel = 128,
                        name = "convblock_1")
            self.network['detection_scale_1'] = self.detector_block(
                        self.network['scale_1_out_final'], self.anchor_scale1,
                        name = "detector_scale_1")
        self.network['combined_detectors'] = tf.concat(
                                [self.network['detection_scale_3'],
                                 self.network['detection_scale_2'],
                                 self.network['detection_scale_1']], axis = 1)
#-------------------------------------------------------------------------------
    def get_feature(self):
        return self.network['combined_detectors']

    def get_logits(self):
        return self.network['combined_detectors']

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

    def leaky_relu(self, input):
        """
        Leaky relu with a slope of 0.1 for x<0
        """
        return tf.nn.leaky_relu(input, alpha = 0.1)

    def conv(self,input, kernel_shape, out_channel, strides, name):
        nr_depth = input.get_shape().as_list()[-1]
        kernel = self.get_var(shape = [kernel_shape[0], kernel_shape[1],
                                  nr_depth,out_channel], name = name + "_W")
        self.network[name + "_W"] = kernel
        if strides[1] > 1:
            padding = [[0,0], [1,1], [1,1], [0,0]]
            input = tf.pad(input, padding)
            conv = tf.nn.conv2d(input, kernel, strides, padding='VALID')
        else:
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
                                    momentum= 0.9, epsilon = 1e-05,
                                    beta_initializer = beta,
                                    gamma_initializer = gamma,
                                    moving_mean_initializer = mean,
                                    moving_variance_initializer = variance,
                                    scale=True,
                                    training=self.is_train, name =name+ '_bn')

        activation = self.leaky_relu(batch)
        return activation

    def conv_bias(self, input, kernel_shape, out_channel, strides, name):
        """
        Implements the convolutions operations along with bias (no batch norm
        and no activation)
        Args:
            input: Input tensor
            kernel_shape: Kernel shape
            out_channel: number of kernels to be applied
            strides: strides
        Returns:
            output tensor with conv+bias
        """
        nr_depth = input.get_shape().as_list()[-1]
        kernel = self.get_var(shape = [kernel_shape[0], kernel_shape[1],
                                  nr_depth,out_channel], name = name + "_W")
        biases = self.get_var(shape=[out_channel], name = name + "_b")
        self.network[name + "_W"] = kernel
        self.network[name + "_b"] = biases
        conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        return conv

    def residual_block(self, input, out_channel, strides, name):
        """
        Residual block used in YOLOv3 architecture. Re-uses the conv blocks with
        addition operation for merging the residula conncetion.
        First convolutions is point-wise, second convolution is normal conv
        Args:
            input: Input tensor
            out_channel: Number of output channels in the non-residual layer
                        The number of final output channels will be double
            strides: Strides used for both convolutional blocks.
        Returns:
            A merged tensor. Input and the second conv will be merged with
            element-wise addition
        KeyError: None
        """
        self.network[name+"_out_1"] = self.conv(input, kernel_shape = [1,1],
           out_channel = out_channel, strides = strides, name = name + "/conv1")

        self.network[name+"_out_2"] = self.conv(self.network[name+"_out_1"],
                            kernel_shape = [3,3], out_channel=2*out_channel,
                            strides = strides, name = name + "/conv2")

        return input + self.network[name+"_out_2"]

    def conv_block(self, input, out_channel, name):
        """
        A collection of convolutions used in YOLOv3 architecture. This block
        consists of convolutions of two kinds,1X1 and 3X3 outputting out_channel
        and 2*out_channel respectively in an alternate order.
        Args:
            input: Input tensor
            out_channel: The number of output channel from point-wise conv
        Returns:
            Tensor with same spatial shape and same output channels.
        """
        self.network[name + '_conv_out_1'] = self.conv(input,
                        kernel_shape = [1,1], out_channel = out_channel,
                        strides = [1,1,1,1], name = name + "/conv1")
        self.network[name + '_conv_out_2'] = self.conv(
                    self.network[name + '_conv_out_1'], kernel_shape = [3,3],
                    out_channel = 2*out_channel,  strides = [1,1,1,1],
                    name = name + "/conv2")
        self.network[name + '_conv_out_3'] = self.conv(
                    self.network[name + '_conv_out_2'],kernel_shape = [1,1],
                    out_channel = out_channel, strides = [1,1,1,1],
                    name = name + "/conv3")
        self.network[name + '_conv_out_4'] = self.conv(
                    self.network[name + '_conv_out_3'], kernel_shape = [3,3],
                    out_channel = 2*out_channel,  strides = [1,1,1,1],
                    name = name + "/conv4")
        self.network[name + '_conv_out_5'] = self.conv(
                    self.network[name + '_conv_out_4'],kernel_shape = [1,1],
                    out_channel = out_channel, strides = [1,1,1,1],
                    name = name + "/conv5")
        self.network[name + '_intermediate'] =self.network[name + '_conv_out_5']
        self.network[name + '_conv_out_6'] = self.conv(
                    self.network[name + '_conv_out_5'], kernel_shape = [3,3],
                    out_channel = 2*out_channel,  strides = [1,1,1,1],
                    name = name + "/conv6")
        return self.network[name + '_conv_out_6']

    def upsample(self, input, out_shape):
        """
        This upsampling is done with nearest_neighbour algorithm.
        Learned kernels can also be applied. TODO: check the upsampling with
        learned kernels.
        Args:
            input: Input tensor
            out_shape: The shape (as list) of the intermediate feature.
        Returns:
            Upsamples feature base on the out_shape
        """
        upsampled_height = out_shape[2] # Why height? This looks wrong
        upsamples_width = out_shape[1]
        shape = (upsampled_height, upsamples_width)
        return tf.image.resize_nearest_neighbor(input, shape)

    def detector_block(self, input, anchors, name):
        """
        The detector block consists of a convolution with out_channel==255,
        followed by the rearrangement of the results from the feature map.
        Args:
            input: Input tensor. For the scale 3, these are simply the scale 3
            output feature map. For scale 2 and scale 1, these are merged multi-
            scale contextual features.
        Returns:
            output feature map consisting of smae spatial dimentions but with
            depth dimention demostrating the predicted outputs.
            Depth == nr_prior_box_per_scale*(5 + nr_class)
        """
        self.network[name + '_conv_out'] = self.conv_bias(input,
                kernel_shape = [1,1],
                out_channel = self.nr_achors_per_detector*(5+self.nr_classes),
                strides = [1,1,1,1], name = name + "/conv")
        feature_shape = self.network[name + '_conv_out'].get_shape().as_list()
        grid_shape = feature_shape[1:3]
        shape = (-1, self.nr_achors_per_detector*grid_shape[0]*grid_shape[1],
                                                            self.nr_classes+5)
        reshaped_feature = tf.reshape(self.network[name + '_conv_out'], shape)
        pixels_per_grid = (self.img_height//grid_shape[0],
                                self.img_width//grid_shape[1])
        # Seperate out the prediction quantities to do different transformations
        box_centers, box_shapes, confidence, classes =tf.split(reshaped_feature,
                                        [2, 2, 1, self.nr_classes], axis=-1)
        x = tf.range(grid_shape[0], dtype = tf.float32)
        y = tf.range(grid_shape[1], dtype = tf.float32)
        x_offset, y_offset = tf.meshgrid(x,y)
        x_offset = tf.reshape(x_offset, [-1,1])
        y_offset = tf.reshape(y_offset, [-1,1])
        # Credits to: https://github.com/shahkaran76
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.tile(x_y_offset, [1, self.nr_achors_per_detector])
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
        box_centers = tf.nn.sigmoid(box_centers)
        box_centers = (box_centers + x_y_offset) * pixels_per_grid
        anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
        box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)
        confidence = tf.nn.sigmoid(confidence)
        classes = tf.nn.sigmoid(classes)
        prediction = tf.concat([box_centers, box_shapes,
                                    confidence, classes], axis=-1)
        return prediction

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
#---------------------------------weights for YOLOv3 from official site

def create_variable_dict_from_weight_file(var_list, weights_file):
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)
    ptr = 0
    i = 0
    data_dict = {}
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        if var1.name.find("conv"):
            if (var2.name.find("_bn")!=-1):
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    data_dict.update(
                            {var.name[var.name.find("/")+1:-2]: var_weights})
                    print(var.name, var_weights.shape)
                i += 4
            elif (var2.name.find("_b")!=-1):
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +bias_params].reshape(bias_shape)
                ptr += bias_params
                data_dict.update(
                            {bias.name[bias.name.find("/")+1:-2]: bias_weights})
                print(bias.name, bias_weights.shape)
                i += 1
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # Why this transpose is required?
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            data_dict.update({var1.name[var1.name.find("/")+1:-2]: var_weights})
            print(var1.name, var_weights.shape)
            i += 1
    return data_dict

def get_YOLOv3_weights_from_darknet_weights():
    url = 'https://pjreddie.com/media/files/yolov3.weights'
    project_path = os.getcwd()
    YOLOv3_weight_path = project_path + "/architecture/version5/"
    filename = "yolov3.weights"
    print("Downloading YOLOv3 checkpoints")
    filename = wget.download(url, YOLOv3_weight_path)
    print("Downloaded " + filename)
    #Create the graph temprarily and then delete it after using the var names
    tf.reset_default_graph()
    img = tf.placeholder(tf.float32, shape = (1, 416, 416, 3))
    net = YOLOv3(None, False)
    net.build_model(img)
    variable_list = tf.global_variables()
    weight_dict = create_variable_dict_from_weight_file(variable_list,
                                     os.path.join(YOLOv3_weight_path, filename))
    tf.reset_default_graph()
    print("Creating npz for YOLOv3 weights")
    np.savez(YOLOv3_weight_path + 'YOLOv3_weights.npz', **weight_dict)
    print("Cleanup downloaded weights")
    os.unlink(YOLOv3_weight_path + 'yolov3.weights')

def get_YOLOv3_weights():
    project_path = os.getcwd()
    YOLOv3_weight_path = project_path + "/architecture/version5/"
    if not os.path.isfile(YOLOv3_weight_path + "YOLOv3_weights.npz"):
        get_YOLOv3_weights_from_darknet_weights()
    weight_dict = np.load(YOLOv3_weight_path + "YOLOv3_weights.npz")
    return weight_dict

def create_model(img, is_train = True, pretrained = False):
    if pretrained:
        YOLOv3_weights = get_YOLOv3_weights()
    else:
        YOLOv3_weights = None
    net = YOLOv3(YOLOv3_weights, is_train)
    net.build_model(img)
    return {'feature_in': img,
            'feature_logits': net.get_logits(),
            'feature_out': net.get_feature(),
            'network': net.get_network()}

#---------------------------------TESTING---------------------------------------
def main():
    img = tf.placeholder(tf.float32, shape = (1, 416, 416, 3))
    model = create_model(img, False, True)
    sample_array = np.ones((1,416,416,3))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(model['feature_out'],
                        feed_dict = {model['feature_in']: sample_array})
    print(out.shape)
    print(out)

if __name__ == "__main__":
    main()
