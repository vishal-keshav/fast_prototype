"""
Dataset Preprocessing script
This module contains two main functionalities:
1. Download the dataset from internet (TODO: stream)
2. Pre-process the dataset (if required)
3. View the dataset before feeding it to the network

Downloading the dataset may also mean locating the dataset on the
local disk in an abandoned(or buried) directory. However, in such a
case, copying the dataset in raw folder may not be a wise choice.
Instead, it is recommended to make a soft link for the dataset in
raw directory of dataset folder.

Pre-processing of dataset requires access to raw data and scripts
that does the following:
* Resize (reshape) the the data.
* Apply any augmentation
* Saving it to a format that is easy for tensorflow data-pipeline
If pre-processed dataset is already present in a buried directory,
then either create a softlink or locate that directory.

A viewer that may use multiple libraries to showcase what is being
feeded into the neural network. The viewer may be a part of library
or a web based viewer implemented by user. For viewables, the
pre-processed data is read.

"""

# THIS SCRIPT NEED NOT BE MODIFIED.

import os
def execute(args):
    import dataset.dataset as ds
    project_path = os.getcwd()
    ds_obj = ds.DataSet(args.dataset, project_path)
    if args.download:
        ds_obj.get_data()
    if args.preprocess:
        ds_obj.preprocess_data()
    if args.view:
        # TODO: Implement a stream viewer,
        # viewer currenlty saves temorary images in debug
        ds_obj.view()
