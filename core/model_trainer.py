# Train script
import os
import importlib
import tensorflow as tf

def execute(args):
    import dataset.dataset as ds
    model_module = "architecture.version" + str(args.model) + ".model"
    arch = importlib.import_module(model_module)
    project_path = os.getcwd()
    ds_obj = ds.DataSet(args.dataset, project_path)
    import hyperparameter.hyperparameter as hp
    hp_obj = hp.HyperParameters(args.param, project_path)
    param_dict = hp_obj.get_params()
    img_batch, label_batch = ds_obj.data_provider(param_dict)
    model = arch.create_model(img_batch, param_dict)
    # Before training, create a checkpoint mechanism
    logs_path = project_path + "/checkpoint/checkpoint_" + str(args.model) + "_" + str(args.param) + "_" + args.dataset
    chk_name = os.path.join(logs_path, 'model.ckpt')
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    writer = tf.summary.FileWriter(logs_path)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if tf.train.checkpoint_exists(chk_name):
            saver.restore(sess, chk_name)
            print("session restored")
        else:
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
        # Code after this needs to be modified as per requirements
        # Depends on optimisation method, logging preference
        #labels = tf.placeholder(tf.float32, (None, 10))
        loss = tf.losses.mean_squared_error(labels = label_batch, predictions = model['feature_out'])
        gradient_optimizer = tf.train.GradientDescentOptimizer(0.0005)
        gd_opt = gradient_optimizer.minimize(loss)
        for i in range(10):
            #feed_dict = {}
            #feed_dict[model['feature_in']] = img_batch
            #feed_dict[labels] = label_batch
            l = sess.run([loss])
            print(l)
        save_path = saver.save(sess, chk_name)
