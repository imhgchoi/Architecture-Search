
import os
import numpy as np
import torch
import random
from torchvision import datasets, transforms
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from pycocotools.coco import COCO
# import sys


class Dataset :
	def __init__(self, args):
		self.args = args

		if args.dataset.upper() == 'CIFAR10' :
			input_size, num_classes, dataset = get_CIFAR10(args.data_path, 
														   batch_size=args.batch_size,
														   valid_perc=0.1, 
														   debug=args.debug)
		if args.dataset.upper() == 'COCO' :
			input_size, num_classes, dataset = get_COCO(args.data_path, 
													    batch_size=args.batch_size,
													    debug=args.debug)
		else :
			raise NotImplementedError

		self.input_size = input_size
		self.num_classes = num_classes
		self.train_loader = dataset[0]
		self.valid_loader = dataset[1]
		self.test_loader = dataset[2]



######### CIFAR 10 

def get_CIFAR10(path, batch_size, valid_perc=0.1, debug=False):
	input_size = 32
	num_classes = 10
	normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

	train_transform = transforms.Compose(
		[
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]
	)
	train_dataset = datasets.CIFAR10(path, train=True, transform=train_transform, download=True)
	if debug :
		train_dataset, _ = torch.utils.data.random_split(train_dataset, 
													     [1000, len(train_dataset)-1000])
	tr_size = len(train_dataset)

	val_num = int(tr_size * valid_perc)
	train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, 
																 [tr_size-val_num, val_num])

	test_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			normalize,
		]
	)
	test_dataset = datasets.CIFAR10(path, train=False, transform=test_transform, download=True)
	if debug :
		test_dataset, _ = torch.utils.data.random_split(test_dataset, 
													    [500, len(test_dataset)-500])

	kwargs = {"num_workers": 2, "pin_memory": True}
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1000, shuffle=False, **kwargs)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, **kwargs)

	return input_size, num_classes, (train_loader, valid_loader, test_loader)



######### COCO 2017

def get_COCO(path, batch_size, debug=False):

	class COCO_Dataset(Dataset):
		def __init__(self, root_dir, set_name, resize=300):
			super(Dataset, self).__init__()
			self.root_dir = root_dir
			self.set_name = set_name
			self.resize = resize
			assert set_name in ['train2017', 'val2017', 'test2017']

			self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json')) 

			self.img_id = list(self.coco.imgToAnns.keys())
			# self.ids = self.coco.getImgIds()

			self.coco_ids = sorted(self.coco.getCatIds())  # list of coco labels [1, ...11, 13, ... 90]  # 0 ~ 79 to 1 ~ 90
			self.coco_ids_to_continuous_ids = {coco_id: i for i, coco_id in enumerate(self.coco_ids)}  # 1 ~ 90 to 0 ~ 79
			# int to int
			self.coco_ids_to_class_names = {category['id']: category['name'] for category in
			                                self.coco.loadCats(self.coco_ids)}  # len 80
			# int to string
			# {1 : 'person', 2: 'bicycle', ...}
			'''
			[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
			 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
			 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
			'''

		def __getitem__(self, index, visualize=False):

			# visualize = False

			# --------------------------- load data ---------------------------
			# 1. image_id
			img_id = self.img_id[index]

			# 2. load image
			img_coco = self.coco.loadImgs(ids=img_id)[0]
			file_name = img_coco['file_name']
			file_path = os.path.join(self.root_dir, 'images', self.set_name, file_name)

			# eg. 'D:\\Data\\coco\\images\\val2017\\000000289343.jpg'
			image = Image.open(file_path).convert('RGB')

			# 3. load anno
			anno_ids = self.coco.getAnnIds(imgIds=img_id)  # img id 에 해당하는 anno id 를 가져온다.
			anno = self.coco.loadAnns(ids=anno_ids)        # anno id 에 해당하는 annotation 을 가져온다.

			det_anno = self.make_det_annos(anno)           # anno -> [x1, y1, x2, y2, c] numpy 배열로

			boxes = torch.FloatTensor(det_anno[:, :4])     # numpy to Tensor
			labels = torch.LongTensor(det_anno[:, 4])

			# --------------------------- for transform ---------------------------
			transform_list = ['photo', 'expand', 'crop', 'flip', 'resize']

			zero_to_one_coord = False
			if 'resize' in transform_list:
				zero_to_one_coord = True

			image, boxes, labels = transform(image, boxes, labels, self.set_name, transform_list, self.resize, zero_to_one_coord)

			if visualize:

				# ----------------- visualization -----------------
				mean = np.array([0.485, 0.456, 0.406])
				std = np.array([0.229, 0.224, 0.225])

				# tensor to img
				img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
				img_vis *= std
				img_vis += mean
				img_vis = np.clip(img_vis, 0, 1)

				plt.figure('input')
				plt.imshow(img_vis)

				for i in range(len(boxes)):

					if not zero_to_one_coord:
					    self.resize = 1
					x1 = boxes[i][0] * self.resize
					y1 = boxes[i][1] * self.resize
					x2 = boxes[i][2] * self.resize
					y2 = boxes[i][3] * self.resize

					# print(boxes[i], ':', self.coco_ids_to_class_names[self.coco_ids[labels[i]]])

					# class
					plt.text(x=x1 - 5,
							 y=y1 - 5,
							 s=str(self.coco_ids_to_class_names[self.coco_ids[labels[i]]]),
							 bbox=dict(boxstyle='round4',
									   facecolor=coco_color_array[labels[i]],
									   alpha=0.9))

					# bounding box
					plt.gca().add_patch(Rectangle(xy=(x1, y1),
					                              width=x2 - x1,
					                              height=y2 - y1,
					                              linewidth=1,
					                              edgecolor=coco_color_array[labels[i]],
					                              facecolor='none'))

					plt.show()

			return image, boxes, labels

		def make_det_annos(self, anno):

			annotations = np.zeros((0, 5))
			for idx, anno_dict in enumerate(anno):

				if anno_dict['bbox'][2] < 1 or anno_dict['bbox'][3] < 1:
					continue

				annotation = np.zeros((1, 5))
				annotation[0, :4] = anno_dict['bbox']

				annotation[0, 4] = self.coco_ids_to_continuous_ids[
				    anno_dict['category_id']]  # 원래 category_id가 18이면 들어가는 값은 16
				annotations = np.append(annotations, annotation, axis=0)  # np.shape()

			# transform from [x, y, w, h] to [x1, y1, x2, y2]
			annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
			annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

			return annotations

		def collate_fn(self, batch):
			"""
			:param batch: an iterable of N sets from __getitem__()
			:return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, difficulties, img_name and
			additional_info
			"""
			images = list()
			boxes = list()
			labels = list()

			for b in batch:
				images.append(b[0])
				boxes.append(b[1])
				labels.append(b[2])

			images = torch.stack(images, dim=0)
			return images, boxes, labels

		def __len__(self):
			return len(self.img_id)




	if debug :
		trainset = COCO_Dataset(path, set_name='val2017')
		validset = COCO_Dataset(path, set_name='val2017')
	else :
		trainset = COCO_Dataset(path, set_name='train2017')
		validset = COCO_Dataset(path, set_name='val2017')

	train_loader = DataLoader(trainset,
							  batch_size=1,
							  # collate_fn=trainset.collate_fn,
							  shuffle=False,
							  num_workers=0,
							  pin_memory=True)

	valid_loader = DataLoader(validset,
							  batch_size=1,
							  # collate_fn=trainset.collate_fn,
							  shuffle=False,
							  num_workers=0,
							  pin_memory=True)
	test_loader = None

	input_size = trainset.resize
	num_classes = 20

	return input_size, num_classes, (train_loader, valid_loader, test_loader)



############# augmentation utility functions


def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    Helps to learn to detect smaller objects.
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.
    Note that some objects may be cut out entirely.
    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def flip(image, boxes):
    """
    Flip image horizontally.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0]
    new_boxes[:, 2] = image.width - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def transform(image, boxes, labels, set_name, transform_list, new_size, zero_to_one_coord=True):

	allowed_tf_list = ['photo', 'expand', 'crop', 'flip', 'resize']
	for tf in transform_list:
		assert tf in allowed_tf_list

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	new_image = image
	new_boxes = boxes
	new_labels = labels

	# Skip the following operations for evaluation/testing
	if 'train' in set_name :

		if 'photo' in transform_list:
			# A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
			new_image = photometric_distort(new_image)

		new_image = FT.to_tensor(new_image)

		if 'expand' in transform_list:
			each_img_mean = torch.mean(new_image, (1, 2))
			# Expand image (zoom out)
			if random.random() < 0.5:
				new_image, new_boxes = expand(new_image, boxes, filler=each_img_mean)

		if 'crop' in transform_list:
			new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

		new_image = FT.to_pil_image(new_image)

		if 'flip' in transform_list:
			# Flip image with a 50% chance
			if random.random() < 0.5:
				new_image, new_boxes = flip(new_image, new_boxes)

	if 'resize' in transform_list:
		# Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
		new_image, new_boxes = resize(new_image, new_boxes, (new_size, new_size), zero_to_one_coord)

	# Convert PIL image to Torch tensor
	new_image = FT.to_tensor(new_image)

	# Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
	new_image = FT.normalize(new_image, mean=mean, std=std)

	return new_image, new_boxes, new_labels



def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps  # (n1, n2)

    return intersection / union  # (n1, n2)