"""
Trainer script requires these:
1. A well defined model (as indiciated by model version)
2. A set of hyper-parameters used by the model and training script (indicated by version)
3. A data provider (as indicated by dataset name)

Once, all the above modules are available (after sanity check),
input and outputs are declared, loss is defined, and optimization
algorithm is selected.
Before starting the training, data-record is initilized which will
be available for visualization.

"""
import os
import importlib
import tensorflow as tf
import numpy as np

def get_hyper_parameters(version, project_path):
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(version, project_path)
    param_dict = hp_obj.get_params()
    return param_dict

"""def get_data_provider(dataset, project_path, param_dict):
    data_provider_module_path = "dataset." + dataset + ".data_provider"
    data_provider_module = importlib.import_module(data_provider_module_path)
    dp_obj = data_provider_module.get_obj(project_path)
    img_batch, label_batch = dp_obj.data_provider(param_dict)
    return img_batch, label_batch"""

def get_data_provider(dataset, project_path, param_dict):
    data_provider_module_path = "dataset." + dataset + ".data_provider"
    data_provider_module = importlib.import_module(data_provider_module_path)
    dp_obj = data_provider_module.get_obj(project_path)
    img_batch = tf.placeholder(tf.float32, [None, 28, 28, 1])
    label_batch = tf.placeholder(tf.float32, [None, 10])
    return img_batch, label_batch, dp_obj

def get_model(version, inputs, param_dict):
    model_module_path = "architecture.version" + str(version) + ".model"
    model_module = importlib.import_module(model_module_path)
    model = model_module.create_model(inputs, param_dict)
    return model

def get_logger(keys, notify):
    import debug.logger as lg
    lg_obj = lg.logger(keys, notify)
    return lg_obj


def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accuracy = (100.0 * correctly_predicted) / predictions.shape[0]
    return accuracy

def mk_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def execute(args):
    # Declare hyper-parameters (param_dict is a dictionary of parameters)
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(args.param, project_path)
    # Define the data provider module
    img_batch, label_batch, dp = get_data_provider(args.dataset, project_path, param_dict)
    dp.set_batch(param_dict['BATCH_SIZE'])
    # Construct a model to be trained
    model = get_model(args.model, img_batch, param_dict)
    # Create a checkpoint mechanism #TODO: Will be moved to logger
    logs_path = project_path + "/checkpoint/checkpoint_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    chk_name = os.path.join(logs_path, 'model.ckpt')
    mk_dir(logs_path)
    writer = tf.summary.FileWriter(logs_path)
    saver = tf.train.Saver()
    # Get a logger and notifier
    log_keys = ["loss", "accuracy"]
    logger = get_logger(log_keys, args.notify)
    # Define optimization procedure
    logits = model['feature_logits']
    output_probability = model['feature_out']
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=label_batch, logits=logits))
    gradient_optimizer = tf.train.AdamOptimizer(param_dict['learning_rate'])
    gd_opt = gradient_optimizer.minimize(loss)
    # Start a session
    with tf.Session() as sess:
        if tf.train.checkpoint_exists(chk_name):
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, chk_name)
            print("session restored")
        else:
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)

        for nr_epochs in range(param_dict['NUM_EPOCHS']):
            for i in range(10):
                img_batch_data, label_batch_data = dp.next()
                feed_dict = {img_batch: img_batch_data, label_batch: label_batch_data}
                _, out, l = sess.run([gd_opt,output_probability,loss], feed_dict = feed_dict)
                accu = accuracy(out, label_batch_data)
                log_list = {'loss': l, 'accuracy': accu}
                logger.batch_logger(log_list, i)
            logger.epoch_logger(nr_epochs)
            save_path = saver.save(sess, chk_name)
