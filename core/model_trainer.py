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

def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accuracy = (100.0 * correctly_predicted) / predictions.shape[0]
    return accuracy

def execute(args):
    # Declare hyper-parameters (param_dict is a dictionary of parameters)
    project_path = os.getcwd()
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(args.param, project_path)
    param_dict = hp_obj.get_params()
    # Define the data provider module
    data_provider_module_path = "dataset." + args.dataset + ".data_provider"
    data_provider_module = importlib.import_module(data_provider_module_path)
    dp_obj = data_provider_module.get_obj(project_path)
    img_batch, label_batch = dp_obj.data_provider(param_dict)
    # Construct a model to be trained
    model_module_path = "architecture.version" + str(args.model) + ".model"
    model_module = importlib.import_module(model_module_path)
    model = model_module.create_model(img_batch, param_dict)
    # Create a checkpoint mechanism
    logs_path = project_path + "/checkpoint/checkpoint_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    chk_name = os.path.join(logs_path, 'model.ckpt')
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    writer = tf.summary.FileWriter(logs_path)
    saver = tf.train.Saver()
    # Start a session
    with tf.Session() as sess:
        if tf.train.checkpoint_exists(chk_name):
            saver.restore(sess, chk_name)
            print("session restored")
        else:
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
        logits = model['feature_logits']
        output_probability = model['feature_out']
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=label_batch, logits=logits))
        gradient_optimizer = tf.train.GradientDescentOptimizer(param_dict['learning_rate'])
        gd_opt = gradient_optimizer.minimize(loss)
        for nr_epochs in range(param_dict['NUM_EPOCHS']):
            print("Training for epoch " + str(nr_epochs))
            for i in range(10):
                out, l, label = sess.run([output_probability,loss, label_batch])
                print("Loss:" + str(l))
                print("Accuracy: " + str(accuracy(out, label)))
            save_path = saver.save(sess, chk_name)
