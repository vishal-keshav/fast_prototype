"""

Project Name: Prototype_v1
Author: Vishal Keshav
Based on: Self
Email: vishal.keshav.1993@gmail.com
External link: None

"""

from core import dataset_preprocessing as dp
from core import model_trainer as mt
#from core import evaluator as e
#from core import generate_visualization as gv
import argparse

# Program meta-data
program_name = "Prototype_v1"
program_details = "A systematic ML project prototyping"

def argument_parser():
    parser = argparse.ArgumentParser(description=program_name + " : " + program_details)
    parser.add_argument('--phase', default='all', type=str, help='Phase')
    parser.add_argument('--dataset', default='MNIST', type=str, help='Dataset')
    parser.add_argument('--download', default=False, type=bool, help='Download dataset?')
    parser.add_argument('--preprocess', default=False, type=bool, help='Preprocess dataset?')
    parser.add_argument('--model', default=1, type=int, help='Model Version')
    parser.add_argument('--param', default=1, type=int, help='Hyper-parameter Version')
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    if args.phase == 'dataset' or args.phase == 'all':
        dp.execute(args)
    if args.phase == 'train' or args.phase == 'all':
        mt.execute(args)
    if args.phase == 'evaluate' or args.phase == 'all':
        e.execute(args)
    if args.phase == 'visualize' or args.phase == 'all':
        gv.execute(args)

if __name__ == "__main__":
    main()
