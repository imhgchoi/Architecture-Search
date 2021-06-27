
import torch
import pdb

class FPNtrainer :
	def __init__(self, args, logger, data, model):
		self.args = args
		self.name = args.name
		self.experiment_name = args.save_name
		self.device = args.device

		# trainee
		self.model = model

	
		# constants 
		self.epochs = args.epochs

		# datasets
		self.data = data
		self.tr_loader = data.train_loader
		self.vl_loader = data.valid_loader
		self.te_loader = data.test_loader

	def train(self):
		

		for epoch in range(1, self.epochs+1):
			for b_idx, (images, boxes, labels) in enumerate(self.tr_loader):
				images = images.to(self.device)
				boxes = [b.to(self.device) for b in boxes]
				labels = [l.to(self.device) for l in labels]
				pdb.set_trace()


        # images = datas[0]
        # boxes = datas[1]
        # labels = datas[2]

        # images = images.to(device)
        # boxes = [b.to(device) for b in boxes]
        # labels = [l.to(device) for l in labels]

        # pred = model(images)
        # loss, (loc, cls) = criterion(pred, boxes, labels)

        # # sgd
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # toc = time.time()