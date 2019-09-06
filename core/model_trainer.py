"""
Trainer script requires these:
1. A well defined model (as indiciated by model version)
2. A set of hyper-parameters used by the model and training script (indicated
    by version)
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

#---------------------------Utility Functions-----------------------------------
def get_hyper_parameters(version, project_path):
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(version, project_path)
    param_dict = hp_obj.get_params()
    return param_dict

def get_data_provider(dataset, project_path, param_dict):
    data_provider_module_path = "dataset." + dataset + ".data_provider"
    data_provider_module = importlib.import_module(data_provider_module_path)
    dp_obj = data_provider_module.get_obj(project_path)
    return dp_obj

def get_model(version, inputs, param_dict, is_train = False, pretrained =False):
    model_module_path = "architecture.version" + str(version) + ".model"
    model_module = importlib.import_module(model_module_path)
    model = model_module.create_model(inputs, is_train, pretrained)
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
        gd_opt_op_grp = tf.group([update_ops, gd_opt_op])
    return loss_op, gd_opt_op_grp

def get_logger(keys, notify):
    import debug.logger as lg
    lg_obj = lg.logger(keys, notify)
    return lg_obj

def accuracy(predictions, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1),
                                      tf.argmax(labels,1))
        accuracy = 100.0*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def correct_op(predictions, labels):
    with tf.name_scope('nr_correct'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1),
                                      tf.argmax(labels,1))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        total_predction = tf.shape(predictions)[0]
    return accuracy, total_predction

def mk_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

#-------------------------------------------------------------------------------

def execute(args):
    tf.random.set_random_seed(47) # Agent 47
    project_path = os.getcwd()
    param_dict = get_hyper_parameters(args.param, project_path)
    dp = get_data_provider(args.dataset, project_path, param_dict)
    train_img, train_label = dp.get_train_dataset(param_dict['BATCH_SIZE'])
    train_model = get_model(args.model, train_img, param_dict, is_train = True,
                                                            pretrained = False)
    valid_img, valid_label = dp.get_validation_dataset(batch_size = 128)
    validation_model = get_model(args.model, valid_img, param_dict,
                                            is_train = False,pretrained = False)

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

    logits = train_model['feature_logits']
    output_probability = train_model['feature_out']
    loss_op, gd_opt_op = optimisation(train_label, logits, param_dict)

    accuracy_op = accuracy(output_probability, train_label)
    nr_correct, nr_total=correct_op(validation_model['feature_out'],valid_label)

    summary_op = tf.summary.merge_all()
    summary_path = project_path + "/debug/summary_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_path + '/train',sess.graph)
        validation_writer = tf.summary.FileWriter(summary_path + '/validation')
        if tf.train.checkpoint_exists(chk_name):
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, chk_name)
            print("session restored")
        else:
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
            print("new session will be saved")

        for nr_epochs in range(param_dict['NUM_EPOCHS']):
            i=0
            dp.initialize_train(sess)
            while True:
                try:
                    _, loss, accu, summary = sess.run([gd_opt_op, loss_op,
                                                      accuracy_op, summary_op])
                    #summary = sess.run(summary_op)
                    train_writer.add_summary(summary,
                                          nr_epochs*param_dict['BATCH_SIZE']+i)
                    i = i+1
                    print("Loss: " + str(loss) + " Examples processed: " + \
                                str(param_dict['BATCH_SIZE']*i)+" " + str(accu))

                    #log_list = {'loss': loss, 'accuracy': accu}
                    #logger.batch_logger(log_list, i)
                except tf.errors.OutOfRangeError:
                    print("Epoch " + str(nr_epochs) + " completed")
                    dp.initialize_validation(sess)
                    correct_examples = 0
                    examples = 0
                    while True:
                        try:
                            correct, total = sess.run([nr_correct, nr_total])
                            correct_examples = correct_examples + correct
                            examples = examples + total
                        except tf.errors.OutOfRangeError:
                            accu_valid = float(correct_examples)/float(examples)
                            print("Validation accuracy: "+str(100.0*accu_valid))
                            break
                    break

            #summary = sess.run(summary_op)
            #validation_writer.add_summary(summary, nr_epochs)
            #logger.epoch_logger(nr_epochs)
            save_path = saver.save(sess, chk_name)
        train_writer.close()
        validation_writer.close()
