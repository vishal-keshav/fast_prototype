"""
Profiler profiles the memory and computation requirements in detail
"""

import tensorflow as tf
import numpy as np

class profiler:
    def __init__(self, graph, verbose = True):
        self.graph = graph
        self.verbose = verbose

    def profile_param(self):
        run_meta = tf.RunMetadata()
        profile_op = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(self.graph, run_meta=run_meta, cmd='op',
                                        options=profile_op)
        if self.verbose:
            print("Total parameters in the graph: " + str(params.total_parameters))
        else:
            # Print in the file
            pass

    def profile_flops(self):
        run_meta = tf.RunMetadata()
        profile_op = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(self.graph, run_meta=run_meta, cmd='op',
                                        options=profile_op)
        if self.verbose:
            print("Total floating point operation in the graph: ", str(flops.total_float_ops))
        else:
            # Print in the file
            pass

    def profile_nodes(self):
        ops_list = self.graph.get_operations()
        tensor_list = np.array([ops.values() for ops in ops_list])
        if self.verbose == True:
            print('PRINTING OPS LIST WITH FEED INFORMATION')
            for t in tensor_list:
                print(t)
        # Iterate over trainable variables, and compute all dimentions
        total_dims = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape() # of type tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_dims += variable_parameters
        if self.verbose == True:
            print('TOTAL DIMS OF TRAINABLE VARIABLES', total_dims)
        else:
            # Print in the file
            pass
