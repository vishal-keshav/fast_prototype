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
    with tf.Session() as sess:
        model = arch.create_model(img_batch, param_dict)
        # Code after this needs to be modified as per requirements
        # Depends on optimisation method, logging preference
        #labels = tf.placeholder(tf.float32, (None, 10))

        loss = tf.losses.mean_squared_error(labels = label_batch, predictions = model['feature_out'])
        gradient_optimizer = tf.train.GradientDescentOptimizer(0.05)
        gd_opt = gradient_optimizer.minimize(loss)
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            #feed_dict = {}
            #feed_dict[model['feature_in']] = img_batch
            #feed_dict[labels] = label_batch
            l = sess.run([loss])
            print(l)
