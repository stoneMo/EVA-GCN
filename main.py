#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo'] = import_class('processor.demo.Demo')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()   # processor.start()
 

# parser is at processor.py

# TRAIN 
# python main.py recognition -c config/st_gcn/kinetics-skeleton/train.yaml --work_dir results/kinetics-skeleton --debug

# TEST
# python main.py recognition -c config/st_gcn/kinetics-skeleton/test.yaml
