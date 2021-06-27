import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import util

from config import Configuration
from dataset import Dataset

from models import ENAS, FPN #import ENAScontroller, ENAStrainer


def load_controller_and_trainer(args, logger, data):
	if args.name == 'ENAS' :
		controller = ENAS.ENAScontroller(args)
		trainer = ENAS.ENAStrainer(args, logger, data, controller)
	elif args.name == 'FPN' :
		model = FPN.FPN(args)
		trainer = FPN.FPNtrainer(args, logger, data, model)
		controller = None
	else :
		raise NotImplementedError

	return controller, trainer


def main():

	args = Configuration()
	util.set_paths(args)
	util.set_seeds(args)
	logger = util.set_logger(args)

	dataset = Dataset(args)
	controller, trainer = load_controller_and_trainer(args=args, 
													  logger=logger, 
													  data=dataset)
	if args.fixed_train :
		trainer.fixed_train()
	else :
		trainer.train()


if __name__ == '__main__' :
	main()