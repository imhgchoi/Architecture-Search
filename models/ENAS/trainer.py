import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

from models.ENAS.child import ENASchild
from util import *

class ENAStrainer :
	def __init__(self, args, logger, data, controller):
		self.args = args
		self.name = args.name
		self.experiment_name = args.save_name
		self.device = args.device

		# constants 
		self.controller_max_episodes = args.controller_max_episodes
		self.controller_iters_per_episode = args.controller_iters_per_episode
		self.child_epochs_per_episode = args.child_epochs_per_episode

		self.controller_lr = args.controller_lr
		self.controller_step_freq = args.controller_step_freq
		self.child_max_lr = args.child_lr_max
		self.child_min_lr = args.child_lr_min
		self.child_T0 = args.child_T0
		self.child_T_mul = args.child_T_mul
		self.child_weight_decay = args.child_weight_decay
		self.child_momentum = args.child_momentum

		self.entropy_constant = args.entropy_weight
		self.baseline_decay = args.bl_decay

		# datasets
		self.data = data
		self.tr_loader = data.train_loader
		self.vl_loader = data.valid_loader
		self.te_loader = data.test_loader

		# trainees
		self.controller = controller
		self.child = ENASchild(args)

		# optimization
		self.controller_optimizer = Adam(
			params=self.controller.parameters(),
			lr=self.controller_lr,
			betas=(0.1, 0.999),
			eps=1e-3
		)
		self.child_optimizer = SGD(
			params=self.child.parameters(),
			lr=self.child_max_lr,
			nesterov=True,
			momentum=self.args.child_momentum,
			weight_decay=self.child_weight_decay
		)
		self.scheduler = CosineAnnealingWarmRestarts(
			optimizer=self.child_optimizer,
			T_0=self.child_T0,
			T_mult=self.child_T_mul,
			eta_min=self.child_min_lr
		)
		self.criterion = nn.CrossEntropyLoss()

		# reporters
		self.writer = SummaryWriter(f'./save/{args.name}/runs/')
		self.logger = logger
		self.save_path = args.model_path

		# initializations
		self.start_epi = 1
		self.start_epoch = 1
		self.baseline = None
		self.best_vl_acc = 0
		self.start_time = time.time()

		if not args.reset :
			self.load(best=args.fixed_train)


	def train(self):

		for epi in range(self.start_epi, self.controller_max_episodes+1) :

			print('\n\n\n')
			self.logger.info(f'#### {self.name} EPISODE {epi} ####')

			loss, acc, reward, adv = self.train_controller()

			arch, _, _ = self.controller()
			self.train_child(arch)
			vl_acc = self.evaluate_child(arch, mode='valid')
			te_acc = self.evaluate_child(arch, mode='test')

			self.start_epi += 1
			if vl_acc > self.best_vl_acc :
				self.save(best=True)
			self.save()

			self.writer.add_scalar('controller mean loss', loss, epi)
			self.writer.add_scalar('controller mean accuracy', acc, epi)
			self.writer.add_scalar('reward', reward, epi)
			self.writer.add_scalar('advantage', adv, epi)
			self.writer.add_scalar('valid accuracy', vl_acc, epi)
			self.writer.add_scalar('test accuracy', te_acc, epi)


	def train_child(self, arch):

		self.child.train()

		stop_epoch = self.start_epoch + self.child_epochs_per_episode
		for epoch in range(self.start_epoch, stop_epoch) :
			print()
			self.logger.info(f'---- CHILD EPOCH {epoch} ----')

			accMeter = AverageMeter()
			lossMeter = AverageMeter()

			for X, y in tqdm(self.tr_loader) :
				X = X.to(self.device)
				y = y.to(self.device)

				self.child.zero_grad()
				pred, _ = self.child(X, arch)
				loss = self.criterion(pred, y)
				loss.backward()
				self.child_optimizer.step()
				tr_acc = torch.sum(y.long()==torch.argmax(pred,dim=1)).float()/y.shape[0]
				accMeter.update(tr_acc.item(), y.shape[0])
				lossMeter.update(loss.item(), y.shape[0])
			self.scheduler.step()

			self.logger.info('TRAIN LOSS \t {0:.4f}'.format(lossMeter()))
			self.logger.info('TRAIN ACCURACY \t {0:.4f}'.format(accMeter()))

			t_msg = elapsed_time(start=self.start_time)
			self.logger.info(t_msg)

		self.start_epoch = stop_epoch


	def train_controller(self):

		self.controller.train()

		accMeter = AverageMeter()
		lossMeter = AverageMeter()
		rewardMeter = AverageMeter()
		advantageMeter = AverageMeter()

		self.controller_optimizer.zero_grad()

		for it in range(1, self.controller_iters_per_episode+1) :
			print('\n\n')
			self.logger.info(f'==== CONTROLLER ITER {it} / {self.controller_iters_per_episode} ====')
			arch, log_prob, entropy = self.controller()
			vl_acc = self.evaluate_child(arch)
			log_prob = torch.sum(log_prob)

			reward = vl_acc + self.entropy_constant * entropy
			if self.baseline is None :
				self.baseline = reward
			self.baseline -= (1-self.baseline_decay) * (self.baseline - reward)
			advantage = reward - self.baseline

			loss = (log_prob * advantage) / self.controller_step_freq
			loss.backward(retain_graph=True)

			if it % self.controller_step_freq == self.controller_step_freq-1 :
				self.controller_optimizer.step()
				self.controller_optimizer.zero_grad()
				print()
				self.logger.info('checkpoint : updated controller params')

			accMeter.update(vl_acc)
			lossMeter.update(loss.item())
			rewardMeter.update(reward.item())
			advantageMeter.update(advantage.item())

		print()
		self.logger.info('CONTROLLER AVG LOSS \t {0:.4f}'.format(lossMeter()))
		self.logger.info('CONTROLLER AVG ACCURACY \t {0:.4f}'.format(accMeter()))
		self.logger.info('CONTROLLER AVG REWARD \t {0:.4f}'.format(rewardMeter()))
		self.logger.info('CONTROLLER AVG ADV \t {0:.4f}'.format(advantageMeter()))

		t_msg = elapsed_time(start=self.start_time)
		self.logger.info(t_msg)

		return lossMeter(), accMeter(), rewardMeter(), advantageMeter()


	def evaluate_child(self, arch, mode='valid'):

		self.child.eval()

		if mode == 'valid' :
			loader = self.vl_loader
		elif mode == 'test' :
			loader = self.te_loader

		accMeter = AverageMeter()
		lossMeter = AverageMeter()

		print()
		self.logger.info(f'---- CHILD EVALUATION : {mode.upper()} MODE ----')

		with torch.no_grad() :
			for X, y in loader :
				X = X.to(self.device)
				y = y.to(self.device)
				pred, _ = self.child(X, arch)
				loss = self.criterion(pred, y)
				vl_acc = torch.sum(y.long()==torch.argmax(pred,dim=1)).float()/y.shape[0]
				accMeter.update(vl_acc.item(), y.shape[0])
				lossMeter.update(loss.item(), y.shape[0])
		self.logger.info(mode.upper()+' LOSS \t {0:.4f}'.format(lossMeter()))
		self.logger.info(mode.upper()+' ACCURACY \t {0:.4f}'.format(accMeter()))

		t_msg = elapsed_time(start=self.start_time)
		self.logger.info(t_msg)

		return accMeter()

	def fixed_train(self):

		def reset_optimizers():
			# reset optimization schemes
			self.child_optimizer = torch.optim.SGD(params=self.child.parameters(),
												   lr=0.05,
												   nesterov=True,
												   momentum=0.9,
												   weight_decay=5e-4)
			self.scheduler = CosineAnnealingLR(optimizer=self.child_optimizer,
											   T_max=10,
											   eta_min=0.001)

		self.start_time = time.time()
		
		# sample best architecture
		print('\n\n')
		self.logger.info('SAMPLING ARCHITECTURE...')
		self.child.train()
		best_arch, best_arch_acc = None, 0 
		for _ in range(10): 
			arch, _, _ = self.controller()

			self.child.reset_params()
			reset_optimizers()
			self.start_epoch = 1
			self.train_child(arch)

			vl_acc = self.evaluate_child(arch, mode='valid')
			if vl_acc > best_arch_acc :
				best_arch = arch
				best_arch_acc = vl_acc
		print('\n\n')
		self.logger.info(f'BEST CONV ARCH = {best_arch[0]}')
		self.logger.info(f'BEST REDUCT ARCH = {best_arch[1]}')

		# train sampled architecture
		print('\n\n')
		self.logger.info('START FIXED TRAIN')
		self.child.reset_params()
		reset_optimizers()

		best_vl_acc, te_acc = 0, None
		for epoch in range(1, 1001): 
			print()
			self.logger.info(f'---- CHILD EPOCH {epoch} ----')

			accMeter = AverageMeter()
			lossMeter = AverageMeter()

			self.child.train()
			for X, y in tqdm(self.tr_loader) :
				X = X.to(self.device)
				y = y.to(self.device)

				self.child.zero_grad()
				pred, _ = self.child(X, best_arch)
				loss = self.criterion(pred, y)
				loss.backward()
				self.child_optimizer.step()

				tr_acc = torch.sum(y.long()==torch.argmax(pred,dim=1)).float()/y.shape[0]
				accMeter.update(tr_acc.item(), y.shape[0])
				lossMeter.update(loss.item(), y.shape[0])
			self.scheduler.step()

			self.logger.info('TRAIN LOSS \t {0:.4f}'.format(lossMeter()))
			self.logger.info('TRAIN ACCURACY \t {0:.4f}'.format(accMeter()))
			t_msg = elapsed_time(start=self.start_time)
			self.logger.info(t_msg)
			
			vl_acc = self.evaluate_child(best_arch, mode='valid')

			if vl_acc > best_vl_acc :
				best_vl_acc = vl_acc
				te_acc = self.evaluate_child(best_arch, mode='test')
				self.save(fixed=True)

			self.logger.info('VALID ACCURACY \t {0:.4f}'.format(vl_acc))
			self.logger.info('BEST STATE VALID ACCURACY \t {0:.4f}'.format(best_vl_acc))
			self.logger.info('BEST STATE TEST ACCURACY \t {0:.4f}'.format(te_acc))


	def save(self, best=False, fixed=False):
		save_dict = {
			'args' : self.args,
			'controller_state_dict' : self.controller.state_dict(),
			'child_state_dict' : self.child.state_dict(),
			'controller_optimizer_state_dict' : self.controller_optimizer.state_dict(),
			'child_optimizer_state_dict' : self.child_optimizer.state_dict(),
			'scheduler_state_dict' : self.scheduler.state_dict(),
			'baseline' : self.baseline,
			'start_epi' : self.start_epi,
			'start_epoch' : self.start_epoch,
			'best_vl_acc' : self.best_vl_acc,
			'start_time' : self.start_time,
			'save_time' : time.time()
		}
		file_path = self.save_path + f'{self.name}' + '_'
		if self.experiment_name != '' :
			file_path += f'_{self.experiment_name}'
		if best :
			file_path += '_best'
		if fixed :
			file_path += '_fixed'
		file_path += '.tar'
		torch.save(save_dict, file_path)


	def load(self, best=False, fixed=False) :
		file_path = self.save_path + f'{self.name}' + '_'
		if self.experiment_name != '' :
			file_path += f'_{self.experiment_name}'
		if best :
			file_path += '_best'
		if fixed :
			file_path += '_fixed'
		file_path += '.tar'
		save_dict = torch.load(file_path)

		self.args = save_dict['args']
		self.controller.load_state_dict(save_dict['controller_state_dict'])
		self.child.load_state_dict(save_dict['child_state_dict'])
		self.controller_optimizer.load_state_dict(save_dict['controller_optimizer_state_dict'])
		self.child_optimizer.load_state_dict(save_dict['child_optimizer_state_dict'])
		self.scheduler.load_state_dict(save_dict['scheduler_state_dict'])
		self.baseline = save_dict['baseline']
		self.start_epi = save_dict['start_epi']
		self.start_epoch = save_dict['start_epoch']
		self.start_epoch = save_dict['start_epoch']
		self.best_vl_acc = save_dict['best_vl_acc']
		
		previously_elapsed_time = save_dict['save_time'] - save_dict['start_time']
		self.start_time = time.time() - previously_elapsed_time