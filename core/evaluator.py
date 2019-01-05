"""
Evaluation of a trained model includes:
1. Variance-bias tradeoff. Showing the training accuracy/loss and validation accuracy/loss with respect to time
2. Different accuracy metrics for test dataset (example: confusion matrix)
3. Other use-case dependent intermediate outputs such as weights and feature outputs.
"""

import tensorflow as tf
from tensorboard import default
from tensorboard import program
import os

class TensorBoard:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def run(self):
        tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
        tb.configure(argv=[None, '--logdir', self.dir_path])
        url = tb.launch()
        print('TensorBoard at %s \n' % url)

def execute(args):
    project_path = os.getcwd()
    summary_dir = project_path + "/debug/summary_" + str(args.model) + \
                "_" + str(args.param) + "_" + args.dataset
    tb_obj = TensorBoard(summary_dir)
    tb_obj.run()
    programPause = raw_input("Press the <ENTER> key to move on.")
