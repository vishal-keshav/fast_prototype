

# A more systematic and advanced approach to prototype research ideas

This repository contains an easy-to-use and easy-to-follow code base :octocat: that aids in fast prototyping machine learning research ideas. The code base relies on the TensorFlow framework, but the same coding style can be incorporated with any ML library such as py-torch and caffe. Each file in this project has been written by keeping one thing in mind: **multiple experimentations with several revisions of ML models and several revisions of hyper-parameters**. Thus, this code base is an initial step to manage the experiments in a more systematic way and track the multiple result output with ample amount of variations and visualizations. :muscle:

# Some key features
1. Bayesian optimizer for hyper-parameter search before you start your training.
2. Inclusion of ready to use popular dataset processing scripts (such as MNIST, CIFAR, IMAGENET).
3. Takes full advantage of TensorFlow's fast data pipeline and efficient GPU computation algorithms.
4. Get notifications of the training progress on your mobile phone.
5. Highly extensible and easy to modify source code.
6. Extensive guide and comments through out the project to guide you through the implementation of a new feature
7. Interesting visualizations such as distribution plots of intermediate layers of neural networks

# Upcoming features
1. Support for training detection models, with more modularity.

## Dependencies
`numpy` `ntfy` `slackclient` `ntfy[slack]` `tensorflow` or `tensorflow_gpu` `keyboard` `opencv_python` `netron` `wget` `pathlib` `statistics` `argparse` `scikit-optimize` `matplotlib` `scipy` `Pillow` `six`

Tested on python version 2 and 3  
All packages can be updated/installed with `pip install -r requirements.txt` except `ntfy[slack]`. Install `ntfy[slack]` seperately with `pip install ntfy[slack]` post requirements.txt installation.

## About the code-base
The main controller is *master.py*. There are two parts which needs further explaination.
* How to use this framework in your own project.
* Where should you make changes.

## How to make use of this repo
Fork the whole repository or make a copy of it. In order to stay updated, :star2::star2:**star this repository**:star2::star2:, and you will be notified :calling: of new versions. Install the dependencies and update them as described in requirements.txt file.

Once the repository is copied :floppy_disk:, interact **only** with *master.py*.
"master.py" is the main controller and it handles following:
* **Phase 1:** Get ready with benchmark datasets
    * Dataset download
    * Data pre-processing
    * Data provider as required in training phase.
* **Phase 2:** Declaring the architecture and training details
    * Model architecture design
    * Hyper-parameter settings
    * Setup intermediate results saving infrastructure
* **Phase 3:** Training and evaluation
    * Training
    * Testing and setting visualization infrastructure
* **Phase 4:** Record results
    * Create tabular results
    * Visualization

### Options for *master.py*

* Dataset Phase options
> python master.py --phase "dataset" --dataset "CIFAR-10" --download True --preprocess True

* Training Phase options
> python master.py --phase "train" --dataset "CIFAR-10" --model 1 --param 1

* Evaluation Phase options
> python master.py --phase "evaluate" --dataset "CIFAR-10" --model 1 --param 1

* Visualization options
> python master.py --phase "visualize" --model 1 --param 1

* No Option - Default - Runs through all of the above phases
* Downloads and preprocess the default MNIST dataset, trains on a default LeNet architecture, evaluate the training, and save the visualizations
> python master.py

## Where to make changes
* dataset/*dataset_name*/dataset_script.py
```
Here, define the methods to retrieve, download and process the dataset.
Two folders, 'raw' and 'pre-processed', shall be used for any pre-processing.
Do not change getter function name and class signature
```

* dataset/*dataset_name*/dataset_provider.py
```
Must define at least two functions, set_batch(batch_size) and next()
Do not change getter function name and class signature
```

* architecture/version_x/model.py
```
Define the architecture in the build function.
Do not change anything else.
For a new model, a new folder with an incremental version number
should be created.
```

* core/model_trainer.py
```
Make changes to get_data_provider() function. Change code only inside
session of execute() function
```

* core/generate_visualization.py
```
Make changes to get_model() function only.
```

* hyperparameter/version_x/param
```
Modify param.json file in a provided version folder or add a new version
folder with a param.json file.
```
