""" Config class for search/augment """
import argparse
import os
import shutil
from functools import partial
import torch

from models.ENAS import get_args
from models.FPN import get_FPN_args


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class Configuration(BaseConfig):
    def build_parser(self):
        parser = get_parser("NAS config")
        parser.add_argument('--name', type=str, default='FPN', choices=['ENAS', 'FPN'],
                            help='name of controller')
        parser.add_argument('--save-name', type=str, default='', help='experiment name')
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--gpus', type=str, default='0', help='gpu device id in str format')
        parser.add_argument('--save-log', action='store_true', default=False)
        parser.add_argument('--debug', action='store_true', default=False)
        parser.add_argument('--flush-runs', action='store_true', default=False)
        parser.add_argument('--flush-logs', action='store_true', default=False)
        parser.add_argument('--seed', type=int, default=1, help='random seed')
        parser.add_argument('--reset', action='store_true', default=False)

        parser.add_argument('--dataset', default='COCO', choices=['CIFAR10', 'COCO'])
        parser.add_argument('--fixed-train', action='store_true', default=False)

        return parser


    def __init__(self):
        parser = self.build_parser()
        base_args = parser.parse_args()

        if base_args.name == 'ENAS' :
            parser = get_args(parser)
        elif base_args.name == 'FPN' :
            parser = get_FPN_args(parser)
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.name = args.name
        self.save_name = args.save_name
        self.path = os.path.join('./', args.name)
        self.logs_path = f'./logs/{args.name}/'
        self.model_path = f'./save/{args.name}/params/'
        # self.gpus = parse_gpus(self.gpus)

        if args.dataset.upper() == 'CIFAR10':
            self.data_path = 'd:/data_repository/CIFAR/cifar10/'
        elif args.dataset.upper() == 'COCO' :
            self.data_path = 'd:/data_repository/COCO/COCO2017/'
        else :
            raise NotImplementedError

        if args.flush_logs :
            shutil.rmtree(f'./logs/{args.name}/')
        if args.flush_runs :
            shutil.rmtree(f'./save/{args.name}/runs')





