"""

Project Name: Prototype_v1
Author: Vishal Keshav
Email: vishal.keshav.1993@gmail.com

"""

from core import debug as d
from core import dataset_preprocessing as dp
from core import hyperparameter_search as hp
#from core import model_trainer as mt
from core import model_trainer_distributed as mt
from core import evaluator as e
from core import generate_visualization as gv
import argparse

# Program meta-data
program_name = "Prototype_v1"
prog_details = "A systematic ML project prototyping"

def argument_parser():
    parser = argparse.ArgumentParser(description=program_name+":"+ prog_details)
    parser.add_argument('--phase', default='all', type=str, help='Phase')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='Dataset')
    parser.add_argument('--download', default=True, type=bool,
                        help='Download dataset?')
    parser.add_argument('--preprocess', default=True, type=bool,
                        help='Preprocess dataset?')
    parser.add_argument('--create_model', default=True, type=bool,
                        help='Create Model from trained data?')
    parser.add_argument('--view', default=False, type=bool,
                        help='View Sample dataset?')
    parser.add_argument('--viewtype', default='model', type=str,
                        help='Type of visualization?')
    parser.add_argument('--notify', default=False, type=bool,
                        help='Training notification?')
    parser.add_argument('--model', default=1, type=int,
                        help='Model Version')
    parser.add_argument('--param', default=1, type=int,
                        help='Hyper-parameter Version')
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    if args.phase == 'debug':
        d.execute(args)
    if args.phase == 'dataset' or args.phase == 'all':
        dp.execute(args)
    if args.phase == 'param_search':
        hp.execute(args)
    if args.phase == 'train' or args.phase == 'all':
        mt.execute(args)
    if args.phase == 'evaluate' or args.phase == 'all':
        e.execute(args)
    if args.phase == 'visualize' or args.phase == 'all':
        gv.execute(args)

if __name__ == "__main__":
    main()
