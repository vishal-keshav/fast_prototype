"""
There are three components for evaluation:
1. Evaluation of dataset (train, test): This require us to generate samples from
the dataset along with the labels, in order to identify if the data feeded to
the network is indeed correct. Image samples are written in debug file, and
labels are printed on the screen.
2. Evaluating the model (train and validation): This require us to visualize
the train and evaluation graph (alone, and combined). We use tensorboard for
this.
3. Evaluating the model on a dataset (both given as an argument): Constructing
the model and then evaluating the train graph and validation graph. For train,
we use a subset of examples, for validation, we use one full epoch.
Evaluating the model on a dataset includes:
1. Evaluating the accuracy.
2. Generating the plot for train-evaliation loss/accuracy.
3. Different accuracy metrics for validation dataset (example: confusion matrix)
4. Other use-case dependent intermediate outputs such as weights and feature
    outputs (here we plot the activation distribution for example).
"""

import tensorflow as tf
from tensorboard import default
from tensorboard import program
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import os
from os import listdir
from os.path import isfile, join
import sys
import importlib
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('seaborn-whitegrid')
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont
from scipy.misc import imread, imresize
#from imagenet_classes import class_names
#------------------------------Utillity Calsses---------------------------------
class TensorBoard:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def run(self):
        tb = program.TensorBoard(default.get_plugins())#,
                                 #default.get_assets_zip_provider())
        tb.configure(argv=[None, '--logdir', self.dir_path])
        url = tb.launch()
        print('TensorBoard at %s \n' % url)

#-------------------------------------------------------------------------------
class op_evaluation:
    def __init__(self, model_ver, param_ver, dataset,project_path,restore=True):
        self.project_path = project_path
        logs_path = project_path + "/checkpoint/checkpoint_" +str(model_ver)+\
                    "_" + str(param_ver) + "_" + dataset
        chk_name = os.path.join(logs_path, 'model.ckpt')
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        if os.path.isdir(logs_path) \
                    and tf.train.checkpoint_exists(chk_name) and restore:
            print("Session restored")
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, chk_name)
        else:
            print("Session cannot be restored")
            self.sess.run(tf.global_variables_initializer())

    def evaluate(self, tf_op, feed_dict = None):
        if feed_dict == None:
            op_out = self.sess.run(tf_op)
        else:
            op_out = self.sess.run(tf_op, feed_dict = feed_dict)
        return op_out

class distribution_plotter:
    def __init__(self):
        self.distribution_data = {}

    def add_distribution_data(self, data, name):
        if name in self.distribution_data.keys():
            self.distribution_data[name] = \
                            np.append(self.distribution_data[name], data)
        else:
            self.distribution_data[name] = data

    def plot_distribution(self, title = 'plot', xlabel = 'xlable',
                                ylabel = 'ylabel', path = ""):
        for key, value in self.distribution_data.items():
            sns.distplot(value, hist = False, kde = True,
                        kde_kws = {'linewidth': 0.6}, label = key)
        plt.figure(1, figsize=(1000,1600))
        #plt.legend(loc=1, prop={'size': 3}, title = title)
        plt.legend(bbox_to_anchor=(1.00,0.5), loc="center left",
                                        prop={'size': 3}, borderaxespad=0)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(path, title + '.jpg'), dpi=600)

class scatter_plotter:
    def __init__(self):
        self.scatter_data = {}

    def add_scatter_data(self, data, name):
        if name in self.scatter_data.keys():
            self.scatter_data[name] = np.vstack([self.scatter_data[name],
                                    np.expand_dims(np.array(data), axis = 0)])
        else:
            self.scatter_data[name] = np.expand_dims(np.array(data), axis=0)

    def plot_scatter(self, title = 'plot', xlabel = 'xlabel',
                    ylabel = 'ylabel', zlabel = 'zlabel', dims = 2, path = ""):
        fig = plt.figure()
        ax = plt.axes()
        for key, value in self.scatter_data.items():
            transformed = self.conditional_pca(self.scatter_data[key])
            im = ax.scatter(transformed['x'], transformed['y'], label = key)
        plt.title(title)
        plt.savefig(os.path.join(path, title + '.jpg'), dpi=600)

    def conditional_pca(self, data):
        pca = PCA(n_components=2)
        pc = pca.fit_transform(data)
        return {'x': pc[:,0], 'y': pc[:,1]}

#----------------------------Utility Function-----------------------------------
def tensorboard_evaluation(args, project_path):
    summary_dir = project_path + "/debug/summary_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    tb_obj = TensorBoard(summary_dir)
    tb_obj.run()
    #For backward compatibility
    if sys.version_info[0] >= 3:
        get_input = input
    else:
        get_input = raw_input
    programPause = get_input("Press the <ENTER> key to move on.")

def get_hyper_parameters(version, project_path):
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(version, project_path)
    param_dict = hp_obj.get_params()
    return param_dict

def get_data_provider(dataset, project_path):
    data_provider_module_path = "dataset." + dataset + ".data_provider"
    data_provider_module = importlib.import_module(data_provider_module_path)
    dp_obj = data_provider_module.get_obj(project_path)
    dp_obj.make_one_shot() #intentionally done snce we dont train here
    return dp_obj

def get_model(version, param_dict, is_train = False, pretrained = True,
                                                            inputs = None):
    model_module_path = "architecture.version" + str(version) + ".model"
    model_module = importlib.import_module(model_module_path)
    if inputs == None:
        inputs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
    model = model_module.create_model(inputs, is_train = is_train,
                                                    pretrained = pretrained)
    return model

def optimisation(label_batch, logits, param_dict):
    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=label_batch, logits=logits))
    tf.summary.scalar('loss', loss_op)
    with tf.name_scope('gradient_optimisation'):
        gradient_opt_op = tf.train.AdamOptimizer(param_dict['learning_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        gd_opt_op = gradient_opt_op.minimize(loss_op)
        gd_opt_op_grp = tf.group([gd_opt_op, update_ops])
    return loss_op, gd_opt_op_grp

def profile_model(args, project_path):
    tf.reset_default_graph()
    #TODO: Fix the hardcoding of shape. default should come from model
    with tf.name_scope('input'):
        img_batch = tf.placeholder(tf.float32, [1, 416, 416, 3])
    param_dict = get_hyper_parameters(args.param, project_path)
    model = get_model(args.model, param_dict, False, False, img_batch)
    graph = tf.get_default_graph()
    import debug.profiler as p
    profile_obj = p.profiler(graph)
    profile_obj.profile_param()
    profile_obj.profile_flops()
    profile_obj.profile_nodes()

def export_model(args, project_path, input_arrays = ["input"],
                                                output_arrays = ["output"]):
    tf.reset_default_graph()
    if not args.create_model:
        return
    logs_path = project_path + "/checkpoint/checkpoint_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    chk_name = os.path.join(logs_path, 'model.ckpt')
    saver = tf.train.import_meta_graph(chk_name + '.meta', clear_devices=True)
    sess = tf.Session()
    saver.restore(sess, chk_name)
    graph = tf.get_default_graph()
    for op in graph.get_operations():
        print (op.name)
    output_graph_def = graph_util.convert_variables_to_constants(sess,
        graph.as_graph_def(), output_node_names.split(","))
    out_path = project_path + "/result/model_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    graph_io.write_graph(output_graph_def, out_path,'model_pb.pb',as_text=False)
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph( out_path + \
                                    "/model_pb.pb", input_arrays, output_arrays)
    tflite_model = converter.convert()
    open(out_path + "/model_tflite.tflite", "wb").write(tflite_model)

def predict_label(e, input):
    img_placeholder, label_placeholder = e.get_placeholders()
    model = e.get_model()
    feed_dict = {img_placeholder: input}
    op = model['feature_out']
    return op,feed_dict

def generate_layer_activation(e, input):
    img_placeholder, label_placeholder = e.get_placeholders()
    model = e.get_model()
    feed_dict = {img_placeholder: input}
    op = model['layer_1']
    return op, feed_dict

def plot_activation(feature, path):
    nr_feature = feature.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(nr_feature / n_columns) + 1
    for i in range(nr_feature):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(feature[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.savefig(path + '/vis_plot.jpg')

def evaluation_on_sample(e, args, project_path, all=True):
    sample_path = project_path + "/dataset/" + args.dataset + "/sample/"
    write_path = project_path + "/visualization/vis" + str(args.model) + "_" + \
                                            str(args.param) + "_" + args.dataset
    if not os.path.isdir(write_path):
        os.mkdir(write_path)
    smple_files=[f for f in listdir(sample_path) if isfile(join(sample_path,f))]
    for file in smple_files:
        """input = cv2.imread(sample_path + file, 0)
        input = input.reshape((1,28,28,1))/255.0"""
        input = imread(sample_path + file, mode='RGB')
        input = imresize(input, (224, 224)).reshape(1, 224, 224, 3)
        op_prob, feed_prob = predict_label(e, input)
        prob = e.evaluate(op_prob, feed_prob)[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])
        op, feed_dict = generate_layer_activation(e, input)
        out = e.evaluate(op, feed_dict)
        plot_activation(out, write_path)
        # Not working with python 3
        #programPause = raw_input("Press the <ENTER> key to move on.")

#----------------------------part one- dataset evaluation-----------------------
def evaluate_dataset(dataset, dataset_type = "validation", nr_data_points = 1):
    tf.reset_default_graph()
    project_path = os.getcwd()
    dp_obj = get_data_provider(dataset, project_path)
    if dataset_type == "train":
        img, label = dp_obj.get_train_dataset(batch_size = 1, shuffle = 1)
    else:
        img, label = dp_obj.get_validation_dataset(batch_size=1,shuffle=1)
    with tf.Session() as sess:
        for i in range(nr_data_points):
            img_out, label_out = sess.run([img, bbox, label])
            write_path = os.path.join(project_path, "debug", dataset)
            # Images are in range [0,1]
            cv2.imwrite(write_path + "_" + dataset_type + "_" + str(i) + ".jpg",
                                                                 img_out[0]*255)
            print(label_out)
            print("data " + str(i) + " printed.")

def evaluate_dataset_object_detection(dataset, dataset_type = "validation",
                                                            nr_data_points = 1):
    def draw_bbox(img, bbox):
        def single_draw_bbox(single_img, single_bbox):
            from skimage.draw import polygon_perimeter
            single_bbox = single_bbox*416
            rr, cc = polygon_perimeter([single_bbox[0]-1, single_bbox[0]-1,
                                        single_bbox[2]-1,single_bbox[2]-1],
                                        [single_bbox[1]-1,single_bbox[3]-1,
                                        single_bbox[3]-1, single_bbox[1]-1])
            single_img[rr, cc, :] = 1
            return single_img
        bboxed_img = []
        for i in range(img.shape[0]):
            single_img = img[i]
            single_bboxes = bbox[i]
            for j in range(single_bboxes.shape[0]):
                single_bbox = single_bboxes[j]
                single_img = single_draw_bbox(single_img, single_bbox)
            bboxed_img.append(single_img)
        return np.array(bboxed_img)
    tf.reset_default_graph()
    project_path = os.getcwd()
    dp_obj = get_data_provider(dataset, project_path)
    if dataset_type == "train":
        img, bbox, label = dp_obj.get_train_dataset(batch_size = 1, shuffle = 1)
    else:
        img, bbox, label = dp_obj.get_validation_dataset(batch_size=1,shuffle=1)
    with tf.Session() as sess:
        for i in range(nr_data_points):
            img_out, bbox_out, label_out = sess.run([img, bbox, label])
            write_path = os.path.join(project_path, "debug", dataset)
            img_out = draw_bbox(img_out, bbox_out)
            cv2.imwrite(write_path + "_" + dataset_type + "_" + str(i) + ".jpg",
                                                                 img_out[0]*255)
            print(label_out)
            print("data " + str(i) + " printed.")

#---------------------------part two- model evaluation--------------------------
def evaluate_model(model_version, model_type = "validation"):
    tf.reset_default_graph()
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(model_version, project_path)
    inputs = tf.placeholder(tf.float32, shape = [None, 416, 416, 3])
    if model_type == "train" or model_type == "both":
        train_model = get_model(model_version,param_dict, is_train = True,
                                    pretrained = False, inputs = inputs)
    if model_type == "validation" or model_type == "both":
        validation_model = get_model(model_version,param_dict,is_train = False,
                                    pretrained = False, inputs = inputs)
    with tf.Session() as sess:
        summary_dir = project_path + "/debug/summary_" + str(model_version)
        saver = tf.summary.FileWriter(summary_dir, sess.graph)
    tb_obj = TensorBoard(summary_dir)
    tb_obj.run()
    #For backward compatibility
    if sys.version_info[0] >= 3:
        get_input = input
    else:
        get_input = raw_input
    programPause = get_input("Press the <ENTER> key to move on.")

#------------------ part three 1 - model evaluation on a dataset ---------------
def evaluate_model_on_dataset(model_version, param_ver, dataset,
                            evaluation_type = "validation", nr_data_points = 1):
    tf.reset_default_graph()
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(model_version, project_path)
    dp_obj = get_data_provider(dataset, project_path)
    if evaluation_type == "train":
        img, label = dp_obj.get_train_dataset(batch_size = 1, shuffle = 1)
        model = get_model(model_version, param_dict, is_train = True,inputs=img)
    else:
        img, label = dp_obj.get_validation_dataset(batch_size =512, shuffle = 1)
        model = get_model(model_version, param_dict,is_train = False,inputs=img)
    # Create an evaluator module, that can evaluate any operation
    evaluator = op_evaluation(model_version, param_ver, dataset, project_path)
    accuracy_op=tf.equal(tf.argmax(model["feature_out"],1), tf.argmax(label, 1))
    accuracy_op = tf.reduce_sum(tf.cast(accuracy_op, tf.float32))
    total_examples = 0.0
    total_nr_correct = 0.0
    while True:
        try:
            # One important point: Dont evaluate the input and output seperately
            # snce it will terate the dataset, making effectvely mismatched result
            [nr_correct,shape]=evaluator.evaluate([accuracy_op,tf.shape(label)])
            total_nr_correct = total_nr_correct + nr_correct
            total_examples = total_examples + float(shape[0])
            print("Evaluated "+ str(total_examples) + " examples")
        except tf.errors.OutOfRangeError:
            print("Evaluation complete")
            break
    print("Accuracy: "+str(100.0*total_nr_correct/total_examples))

def evaluate_model_on_datapoint(model_version, param_ver, dataset,
                                                    file_name = "sample.jpg"):
    tf.reset_default_graph()
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(model_version, project_path)
    dp_obj = get_data_provider(dataset, project_path)
    default_shape = dp_obj.get_input_output_shape()
    img = tf.placeholder(tf.float32, shape = default_shape['input'])
    model = get_model(model_version, param_dict, is_train = False, inputs=img)
    file_path = project_path + "/debug/summary_" + str(model_version) + "_" + \
                                    str(param_ver) + "_" + dataset
    input = imread(os.path.join(file_path,file_name), mode='RGB')
    #TODO: generalize to all default dataset shapes
    #TODO: generalize to other than imagenet
    imagenet_classes_path = "architecture.version" + str(model_version)\
                                                + ".imagenet_classes"
    imagenet_classes = importlib.import_module(imagenet_classes_path)
    class_names = imagenet_classes.class_names
    input = imresize(input, (224, 224)).reshape(1, 224, 224, 3).astype(float)
    input/=127.5
    input-=1.
    feed_dict = {img: input}
    evaluator = op_evaluation(model_version, param_ver, dataset, project_path)
    op_prob = evaluator.evaluate(model['feature_out'], feed_dict = feed_dict)
    preds = (np.argsort(op_prob[0])[::-1])[0:5]
    for p in preds:
        print(class_names[p-1], op_prob[0][p])

def evaluate_object_detection_model_on_datapoint(model_version, param_ver,
                                            dataset, file_name = "sample.jpg"):
    # Utility functions
    def build_boxes(inputs):
        center_x, center_y, width, height, confidence, classes = \
            tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
        # Convert from center/dimention space to box corner coordinates
        top_left_x = center_x - width / 2
        top_left_y = center_y - height / 2
        bottom_right_x = center_x + width / 2
        bottom_right_y = center_y + height / 2
        boxes = tf.concat([top_left_x, top_left_y,
                           bottom_right_x, bottom_right_y,
                           confidence, classes], axis=-1)
        return boxes

    def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                            confidence_threshold):
        batch = tf.unstack(inputs) # Unstack by default on first axis (batch)
        # The unstack reduces the dimention, so now it is matrix 3*13*13X85
        boxes_dicts = []
        for boxes in batch:
            # Reduce 3*13*13 to something very less. Now its 100X85
            boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
            # Now it will give 100 ( the indicies on the class)
            classes = tf.argmax(boxes[:, 5:], axis=-1)
            # Exand so that 100X1
            classes = tf.expand_dims(tf.to_float(classes), axis=-1)
            # Now we have 100X6, where last axis, last entries are class index
            boxes = tf.concat([boxes[:, :5], classes], axis=-1)

            boxes_dict = dict()
            # Now consider class wise iteration, if this class is there in the
            # prediction
            # Instead of if this position contains some class.
            for cls in range(n_classes):
                mask = tf.equal(boxes[:, 5], cls)
                mask_shape = mask.get_shape()
                # Check for if there does not exist a class prediction
                if mask_shape.ndims != 0:
                    # Now 100X6 reduces to 10X6
                    class_boxes = tf.boolean_mask(boxes, mask)
                    # Last entry is not required, we are considering that class
                    boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                                  [4, 1, -1],
                                                                  axis=-1)
                    boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                    # Select only max_output_size boxes
                    indices = tf.image.non_max_suppression(boxes_coords,
                                                           boxes_conf_scores,
                                                           max_output_size,
                                                           iou_threshold)
                    # now we have 2X4 (two boxes X four coordinates) for 1 class
                    class_boxes = tf.gather(class_boxes, indices)
                    boxes_dict[cls] = class_boxes[:, :5]
            # Boxes_dicts has all the boxes for all the batches
            boxes_dicts.append(boxes_dict)
        return boxes_dicts

    def draw_boxes(img_name, boxes_dict, class_names, scaled_size,
                                                        file_path, file_name):
        def draw_bounding_box(im, canvas, coordinates, type = "absolute",
                                                        scaled_size = None):
            im_height = im.size[0]
            im_width = im.size[1]
            if type == "absolute":
                canvas.rectangle(coordinates, outline = (0,0,225,0), width = 2)
            if type == "relative":
                scaling_height = im_height/scaled_size[0]
                scaling_width  = im_width/scaled_size[1]
                scaled_coordinates = [coordinates[0]*scaling_height,
                                      coordinates[1]*scaling_width,
                                      coordinates[2]*scaling_height,
                                      coordinates[3]*scaling_width]
                canvas.rectangle(scaled_coordinates, outline = (0,0,225,0),
                                                                     width = 2)
            return canvas
        img = Image.open(img_name)
        #img = img.resize(scaled_size)
        canvas = ImageDraw.Draw(img)
        for c,b in boxes_dict.items():
            for box_number in range(b.shape[0]):
                canvas = draw_bounding_box(img, canvas, b[box_number],
                                type = "relative", scaled_size = scaled_size)
        img.save(os.path.join(file_path, file_name), "JPEG")

    tf.reset_default_graph()
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(model_version, project_path)
    dp_obj = get_data_provider(dataset, project_path)
    default_shape = dp_obj.get_input_output_shape()
    input_shape = default_shape['input']
    img = tf.placeholder(tf.float32, shape = input_shape)
    model = get_model(model_version, param_dict, is_train = False, inputs=img)
    file_path = project_path + "/debug/summary_" + str(model_version) + "_" + \
                                    str(param_ver) + "_" + dataset
    input = Image.open(os.path.join(file_path,file_name))
    input = np.array(input.resize(input_shape[1:3], Image.ANTIALIAS))
    input = input[np.newaxis,:,:,:]/255.0 # Expanded and normalized
    class_label_path = "architecture/version" + str(model_version)\
                                                + "/coco.names"
    with open(class_label_path, 'r') as f:
        class_names = f.read().splitlines()
    n_classes = 80
    max_output_size = 10
    iou_threshold = 0.5 # for the objectness
    confidence_threshold = 0.5 # for the class probability
    boxes_dicts=non_max_suppression(build_boxes(model["feature_out"]),n_classes,
                        max_output_size, iou_threshold, confidence_threshold)
    feed_dict = {img: input}
    evaluator = op_evaluation(model_version, param_ver, dataset, project_path)
    boxes = evaluator.evaluate(boxes_dicts, feed_dict = feed_dict)
    #for key, value in boxes[0].items():
    #    print(key, value.shape)
    draw_boxes(os.path.join(file_path,file_name), boxes[0], class_names,
                [input_shape[1], input_shape[2]], file_path, "processed.jpg")

#----------------- part three 2 - confusion matrix -----------------------------
def evaluate_model_on_dataset_confusion(model_version, param_ver, dataset,
                            evaluation_type = "validation", nr_data_points = 1):
    tf.reset_default_graph()
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(model_version, project_path)
    dp_obj = get_data_provider(dataset, project_path)
    if evaluation_type == "train":
        img, label = dp_obj.get_train_dataset(batch_size = 1, shuffle = 1)
        model = get_model(model_version, param_dict, is_train = True,inputs=img)
    else:
        img, label = dp_obj.get_validation_dataset(batch_size =512, shuffle = 1)
        model = get_model(model_version, param_dict,is_train = False,inputs=img)
    # Create an evaluator module, that can evaluate any operation
    evaluator = op_evaluation(model_version, param_ver, dataset, project_path)
    confusion_op = tf.confusion_matrix(tf.argmax(label, axis=1),
                                        tf.argmax(model["feature_out"], axis=1))
    confusion_matrix = evaluator.evaluate(confusion_op)
    path = project_path + "/visualization/vis" + str(model_version) + "_" + \
                                    str(param_ver) + "_" + dataset
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.figure(1, figsize=(100,100))
    plt.title('Confusion matrix heat map')
    plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
    plt.savefig(path + '/confusion_matrix.jpg')
    print("Heat-map saved in visualization")

#------- part three 3 - generate the layer and channel output statistics -------
def generate_layer_statistics(model_version, param_ver, dataset,
                        evaluation_type = "validation", nr_data_points = 1):
    # For demostraction, we just show the statistics for the activation from
    # second layer (separate for each channel)
    tf.reset_default_graph()
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(model_version, project_path)
    dp_obj = get_data_provider(dataset, project_path)
    if evaluation_type == "train":
        img, label = dp_obj.get_train_dataset(batch_size = 1, shuffle = 1)
        model = get_model(model_version, param_dict, is_train = True,inputs=img)
    else:
        img, label = dp_obj.get_validation_dataset(batch_size =128, shuffle = 1)
        model = get_model(model_version, param_dict,is_train = False,inputs=img)
    # Create an evaluator module, that can evaluate any operation
    evaluator = op_evaluation(model_version, param_ver, dataset, project_path)

    layer = model['network']['h_conv2']
    dist_plot = distribution_plotter()
    j = 0
    while True: # Go through full validation set
        try:
            j = j+1
            layer_out = evaluator.evaluate(layer)
            for i in range(layer_out.shape[-1]):
                dist_plot.add_distribution_data(
                                    layer_out[:,:,:,i].flatten(),
                                    "h_conv_2_channel_" + str(i))
            if j == 5:
                break
        except tf.errors.OutOfRangeError:
            print("evaluation done")
            break
    path = project_path + "/visualization/vis" + str(model_version) + "_" + \
                                    str(param_ver) + "_" + dataset
    if not os.path.isdir(path):
        os.mkdir(path)
    dist_plot.plot_distribution(title = "conv2_layer_channel_distribution",
                                                                path = path)

def generate_gradient_statistics(model_version, param_ver, dataset,
                        evaluation_type = "validation", nr_data_points = 1):
    # For demostraction, we just show the statistics for the gradients of loss
    # wth respect to weight from second layer (separate for each channel)
    tf.reset_default_graph()
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(model_version, project_path)
    dp_obj = get_data_provider(dataset, project_path)
    if evaluation_type == "train":
        img, label = dp_obj.get_train_dataset(batch_size = 1, shuffle = 1)
        model = get_model(model_version, param_dict, is_train = True,inputs=img)
    else:
        img, label = dp_obj.get_validation_dataset(batch_size =128, shuffle = 1)
        model = get_model(model_version, param_dict,is_train = False,inputs=img)
    logits = model['feature_logits']

    # Create an evaluator module, that can evaluate any operation
    evaluator = op_evaluation(model_version, param_ver, dataset, project_path)
    loss_op, grad_minimization_op =optimisation(label, logits, param_dict)

    weight = model['network']['W_conv2']
    weight_gradient = tf.gradients(loss_op, weight)[0]
    dist_plot = distribution_plotter()
    j = 0
    while True: # Go through full validation set
        try:
            j = j+1
            # First train once, then compute the gradent on updated point
            _, grad= evaluator.evaluate([grad_minimization_op, weight_gradient])
            dist_plot.add_distribution_data(grad.flatten(),
                                "W_conv_2_gradent_" + str(j))
            if j == 5:
                break
        except tf.errors.OutOfRangeError:
            print("evaluation done")
            break
    path = project_path + "/visualization/vis" + str(model_version) + "_" + \
                                    str(param_ver) + "_" + dataset
    if not os.path.isdir(path):
        os.mkdir(path)
    dist_plot.plot_distribution(title = "conv2_weight_gradient_distribution",
                                                                path = path)

def execute(args):
    project_path = os.getcwd()
    #evaluate_dataset(args.dataset, dataset_type = "train", nr_data_points = 5)
    #evaluate_dataset_object_detection(args.dataset)
    #evaluate_model(args.model, model_type = "train")
    #evaluate_model_on_dataset(args.model, args.param, args.dataset)
    #evaluate_model_on_dataset_confusion(args.model, args.param, args.dataset)
    #evaluate_model_on_datapoint(args.model, args.param, args.dataset)
    #evaluate_object_detection_model_on_datapoint(args.model, args.param,
    #                                                            args.dataset)
    #generate_layer_statistics(args.model, args.param, args.dataset)
    #generate_gradient_statistics(args.model, args.param, args.dataset)
    #tensorboard_evaluation(args, project_path)
    #profile_model(args, project_path)
    #create_model(args, project_path)
