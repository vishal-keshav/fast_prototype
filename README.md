# A systematic approach to conduct ML research

This repository contains an easy-to-use heirarchy of code base
in order to conduct Machine Learning research in an efficient way.
Each file has been written keeping in mind the need to experiment with
various ML models, track the results and present it in a good way.

The main controller is "master.py". There are two parts which is needed
to be told before using this code base.
1. How to make use of it.
2. Where to make changes.

## How to make use of it.
Fork the whole repository or make a copy of it. In order to remain updated, star mark this
repository, and you will be notified about newer version. Install the requirements or update
the dependencies.

Once the repository is being copied, interact only with "master.py".
"master.py" handles following:
* Phase 1: Get ready with benchmark dataset(s)
    * Dataset download
    * Data pre-processing
    * Data provider as required in training phase.
* Phase 2: Declararing the architecture(s) and training detail(s)
    * Model architecture design
    * Hyper-parameter settings
    * Setup intermediate results saving infrastructure
* Phase 3: Training and evaluation
    * Training
    * Testing and setting visualization infrastructure
* Phase 4: Record results
    * Create tabular results
    * Visualization

### Options for "master.py"
1. Dataset Phase
example: python master.py --phase "dataset" --dataset "CIFAR-10" --download True --preprocess True
2. Training Phase
example: python master.py --phase "train" --dataset "CIFAR-10" --model 1 --param 1
3. Evaluation Phase
example: python master.py --phase "evaluate" --dataset "CIFAR-10" --model 1 --param 1
4. Visualization
example: python master.py --phase "visualize" --model 1 --param 1
5. Do all of the above
example: python master.py --dataset "CIFAR-10" --download True --preprocess True

## Where to make changes
Will be updated soon.
