# Core-data-processing script
import os
def execute(args):
    import dataset.dataset as ds
    project_path = os.getcwd()
    ds_obj = ds(args['dataset'], project_path)
    if args['download']:
        ds_obj.get_data()
    if args['preprocess']:
        ds_obj.preprocess_data()
    print("Data pre-processing done")
