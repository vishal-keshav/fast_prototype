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
    with tf.name_scope('input'):
        img_batch = tf.placeholder(tf.float32, [None, 28, 28, 1])
    with tf.name_scope('output'):
        label_batch = tf.placeholder(tf.int64, [None, 10])
    return img_batch, label_batch, dp_obj

def get_model(version, inputs, param_dict):
    model_module_path = "architecture.version" + str(version) + ".model"
    model_module = importlib.import_module(model_module_path)
    model = model_module.create_model(inputs, param_dict)
    return model

def optimisation(label_batch, logits, param_dict):
    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=label_batch, logits=logits))
    tf.summary.scalar('loss', loss_op)
    with tf.name_scope('gradient_optimisation'):
        gradient_optimizer_op = tf.train.AdamOptimizer(param_dict['learning_rate'])
        gd_opt_op = gradient_optimizer_op.minimize(loss_op)
    return loss_op, gd_opt_op

def get_logger(keys, notify):
    import debug.logger as lg
    lg_obj = lg.logger(keys, notify)
    return lg_obj

def accuracy(predictions, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels,1))
        accuracy = 100*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
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
    loss_op, gd_opt_op = optimisation(label_batch, logits, param_dict)
    # Define accuracy operation
    accuracy_op = accuracy(output_probability, label_batch)
    # Merge all summaries
    summary_op = tf.summary.merge_all()
    summary_path = project_path + "/debug/summary_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    # Start a session
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_path + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(summary_path + '/validation')
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
                _, out, loss, accu, summary = sess.run([gd_opt_op,output_probability,loss_op, accuracy_op, summary_op], feed_dict = feed_dict)
                train_writer.add_summary(summary, nr_epochs*10 + i)

                img_validation_data, label_validation_data = dp.validation()
                feed_dict = {img_batch: img_validation_data, label_batch: label_validation_data}
                summary = sess.run(summary_op, feed_dict = feed_dict)
                validation_writer.add_summary(summary, nr_epochs*10 + i)
                log_list = {'loss': loss, 'accuracy': accu}
                logger.batch_logger(log_list, i)
            logger.epoch_logger(nr_epochs)
            save_path = saver.save(sess, chk_name)
        train_writer.close()
        validation_writer.close()
