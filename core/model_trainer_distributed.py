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
def construct_train_fn(model, param_dict, is_train = True,pretrained = False):
    def train_step(inputs):
        train_img, train_label = inputs['image'], inputs['label']
        train_model = get_model(model,train_img,param_dict,is_train,pretrained)
        logits = train_model['feature_logits']
        output_probability = train_model['feature_out']
        loss_op, gd_opt_op = optimisation(train_label, logits, param_dict)
        accuracy_op = accuracy(output_probability, train_label)
        with tf.control_dependencies([gd_opt_op]):
            return tf.identity(loss_op), tf.identity(accuracy_op)
    return train_step

def construct_valid_fn(model, param_dict, is_train = True,pretrained = False):
    def validation_step(inputs):
        valid_img, valid_label = inputs['image'], inputs['label']
        valid_model = get_model(model,valid_img,param_dict,is_train,pretrained)
        logits = valid_model['feature_logits']
        output_probability = valid_model['feature_out']
        nr_correct, nr_total=correct_op(valid_model['feature_out'],valid_label)
        return nr_correct, nr_total
    return validation_step
#-------------------------------------------------------------------------------

def execute(args):
    tf.random.set_random_seed(47) # Agent 47
    project_path = os.getcwd()
    #TODO: Change based on args
    strategy = tf.distribute.MirroredStrategy()
    param_dict = get_hyper_parameters(args.param, project_path)
    train_fn = construct_train_fn(args.model, param_dict)
    validation_fn = construct_valid_fn(args.model, param_dict)

    with strategy.scope():
        dp = get_data_provider(args.dataset, project_path, param_dict)
        dp.make_distributed(strategy)
        train_iter = dp.get_train_dataset(param_dict['BATCH_SIZE'])
        rep_loss, accuracy_op = strategy.experimental_run(train_fn, train_iter)
        avg_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, rep_loss)
        accuracy_op = strategy.reduce(tf.distribute.ReduceOp.MEAN, accuracy_op)

        valid_iter = dp.get_validation_dataset(batch_size = 128)
        nr_correct,nr_total=strategy.experimental_run(validation_fn,valid_iter)
        nr_correct = strategy.reduce(tf.distribute.ReduceOp.SUM, nr_correct)
        nr_total = strategy.reduce(tf.distribute.ReduceOp.SUM, nr_total)

        logs_path = project_path + "/checkpoint/checkpoint_"+str(args.model) + \
                    "_" + str(args.param) + "_" + args.dataset
        chk_name = os.path.join(logs_path, 'model.ckpt')
        mk_dir(logs_path)

        writer = tf.summary.FileWriter(logs_path)
        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()
        summary_path = project_path + "/debug/summary_" + str(args.model) + \
                    "_" + str(args.param) + "_" + args.dataset

        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            train_writer = tf.summary.FileWriter(summary_path + '/train',
                                                                    sess.graph)
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
                        loss_aggregate, accu, summary = sess.run([avg_loss,
                                                    accuracy_op, summary_op])
                        train_writer.add_summary(summary,
                                           nr_epochs*param_dict['BATCH_SIZE']+i)
                        i = i+1
                        print("Loss: "+str(loss_aggregate)+" Ex processed: " + \
                                str(param_dict['BATCH_SIZE']*i)+" " + str(accu))
                    except tf.errors.OutOfRangeError:
                        dp.initialize_validation(sess)
                        correct_examples = 0
                        examples = 0
                        while True:
                            try:
                                correct,total = sess.run([nr_correct, nr_total])
                                correct_examples = correct_examples + correct
                                examples = examples + total
                            except tf.errors.OutOfRangeError:
                                ac_valid=float(correct_examples)/float(examples)
                                print("Valid_accuracy:"+str(100.0*ac_valid))
                                break
                        break
                save_path = saver.save(sess, chk_name)
            train_writer.close()
