import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import random
import numbers
import torchvision
import os
import torch.cuda.amp as amp
from model.discriminator import FCDiscriminator
from model.discriminator_dsc import DSCDiscriminator
from model.build_BiSeNet import BiSeNet
import torch.optim as optim
from torch.autograd import Variable


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""

	lr = init_lr*(1 - iter/max_iter)**power
	optimizer.param_groups[0]['lr'] = lr
	return lr
	

def get_label_info(csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	label = {}
	for iter, row in ann.iterrows():
		label_name = row['name']
		r = row['r']
		g = row['g']
		b = row['b']
		class_11 = row['class_11']
		label[label_name] = [int(r), int(g), int(b), class_11]
	return label

def one_hot_it(label, label_info):
	# return semantic_map -> [H, W]
	semantic_map = np.zeros(label.shape[:-1])
	for index, info in enumerate(label_info):
		color = label_info[info]
		equality = np.equal(label, color)
		class_map = np.all(equality, axis=-1)
		semantic_map[class_map] = index
		
	return semantic_map


def one_hot_it_v11(label, label_info):
	# return semantic_map -> [H, W, class_num]
	semantic_map = np.zeros(label.shape[:-1])
	# from 0 to 11, and 11 means void
	class_index = 0
	for index, info in enumerate(label_info):
		color = label_info[info][:3]
		class_11 = label_info[info][3]
		if class_11 == 1:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			semantic_map[class_map] = class_index
			class_index += 1
		else:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			semantic_map[class_map] = 11
	return semantic_map

def one_hot_it_v11_dice(label, label_info):
	# return semantic_map -> [H, W, class_num]
	semantic_map = []
	void = np.zeros(label.shape[:2])
	for index, info in enumerate(label_info):
		color = label_info[info][:3]
		class_11 = label_info[info][3]
		if class_11 == 1:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			semantic_map.append(class_map)
		else:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			void[class_map] = 1
	semantic_map.append(void)
	semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
	return semantic_map

def reverse_one_hot(image):
	"""
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
	w = image.shape[0]
	h = image.shape[1]
	x = np.zeros([w,h,1])
  

	image = image.permute(1, 2, 0)
	x = torch.argmax(image, dim=-1)
	return x


def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	label_values = [label_values[key][:3] for key in label_values if label_values[key][3] == 1]
	label_values.append([0, 0, 0])
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]

	return x

def compute_global_accuracy(pred, label):
	pred = pred.flatten()
	label = label.flatten()
	total = len(label)
	count = 0.0
	for i in range(total):
		if pred[i] == label[i]:
			count = count + 1.0
	return float(count) / float(total)

def fast_hist(a, b, n):
	'''
	a and b are predict and mask respectively
	n is the number of classes
	'''
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
	epsilon = 1e-5
	return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

class RandomCrop(object):
	"""Crop the given PIL Image at a random location.

	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
		pad_if_needed (boolean): It will pad the image if smaller than the
			desired size to avoid raising an exception.
	"""

	def __init__(self, size, seed, padding=0, pad_if_needed=False):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding
		self.pad_if_needed = pad_if_needed
		self.seed = seed

	@staticmethod
	def get_params(img, output_size, seed):
		"""Get parameters for ``crop`` for a random crop.

		Args:
			img (PIL Image): Image to be cropped.
			output_size (tuple): Expected output size of the crop.

		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
		random.seed(seed)
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w
		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return i, j, th, tw

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.

		Returns:
			PIL Image: Cropped image.
		"""
		if self.padding > 0:
			img = torchvision.transforms.functional.pad(img, self.padding)

		# pad the width if needed
		if self.pad_if_needed and img.size[0] < self.size[1]:
			img = torchvision.transforms.functional.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
		# pad the height if needed
		if self.pad_if_needed and img.size[1] < self.size[0]:
			img = torchvision.transforms.functional.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

		i, j, h, w = self.get_params(img, self.size, self.seed)

		return torchvision.transforms.functional.crop(img, i, j, h, w)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

def cal_miou(miou_list, csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	miou_dict = {}
	cnt = 0
	for iter, row in ann.iterrows():
		label_name = row['name']
		class_11 = int(row['class_11'])
		if class_11 == 1:
			miou_dict[label_name] = miou_list[cnt]
			cnt += 1
	return miou_dict, np.mean(miou_list)


class OHEM_CrossEntroy_Loss(nn.Module):
	def __init__(self, threshold, keep_num):
		super(OHEM_CrossEntroy_Loss, self).__init__()
		self.threshold = threshold
		self.keep_num = keep_num
		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def forward(self, output, target):
		loss = self.loss_function(output, target).view(-1)
		loss, loss_index = torch.sort(loss, descending=True)
		threshold_in_keep_num = loss[self.keep_num]
		if threshold_in_keep_num > self.threshold:
			loss = loss[loss>self.threshold]
		else:
			loss = loss[:self.keep_num]
		return torch.mean(loss)

def group_weight(weight_group, module, norm_layer, lr):
	group_decay = []
	group_no_decay = []
	for m in module.modules():
		if isinstance(m, nn.Linear):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
			if m.weight is not None:
				group_no_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)

	assert len(list(module.parameters())) == len(group_decay) + len(
		group_no_decay)
	weight_group.append(dict(params=group_decay, lr=lr))
	weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
	return weight_group

#load and upload model 

def best_model(args, model, model_d, optimizer, optimizer_d, epoch, name = None):
  if name is None:
      filename = args.checkpoint_name_save  
  else:
      filename = args.checkpoint_name_save.replace(".pth", "_" + name + "_" + ".pth") 

  torch.save({
        'optimizer_state': optimizer.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
        'model_state': model.state_dict(),
        'model_d_state': model_d.state_dict(),
        'total_epoch_so_far': epoch
    }, os.path.join(args.save_model_path, filename))

       

def upload_model(args, model, model_d, optimizer, optimizer_d):
   print("Upload Model...")
   path = os.path.join(args.save_model_path, args.checkpoint_name_load)
   checkpoint = torch.load(path)
   print(path)
   optimizer.load_state_dict(checkpoint['optimizer_state'])
   optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
   model.load_state_dict(checkpoint['model_state'])
   model_d.load_state_dict(checkpoint['model_d_state'], strict=False)
   epoch = checkpoint['total_epoch_so_far']
  
   return model, model_d, optimizer, optimizer_d, epoch

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def encode_segmap(label, labelToMap, ignore_index):
    result = ignore_index * np.ones(label.shape, dtype=np.float32)
    for k, v in labelToMap.items():
        result[label == k] = v

    return result

def adjust_learning_rate_D(optimizer, i_iter, lrate, num_steps, power):
    lr = lr_poly(lrate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

palette = [128, 64, 128,  # road, 0
            244, 35, 232,  # sidewalk, 1
            70, 70, 70,  # building, 2
            102, 102, 156,  # wall, 3
            190, 153, 153,  # fence, 4
            153, 153, 153,  # pole, 5
            250, 170, 30,  # traffic light, 6
            220, 220, 0,  # traffic sign, 7
            107, 142, 35,  # vegetation, 8
            152, 251, 152,  # terrain, 9
            70, 130, 180,  # sky, 10
            220, 20, 60,  # person, 11
            255, 0, 0,  # rider, 12
            0, 0, 142,  # car, 13
            0, 0, 70,  # truck, 14
            0, 60, 100,  # bus, 15
            0, 80, 100,  # train, 16
            0, 0, 230,  # motor-bike, 17
            119, 11, 32]  # bike, 18

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
    
def rgb_label(label):
    
    new_mask = Image.fromarray(label.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def gaussian_noise(image):
    noise = torch.zeros_like(image, device = "cuda")
    x = int(np.random.uniform(0, image.shape[2]))
    y = int(np.random.uniform(0, image.shape[3]))
    len_x = int(np.random.uniform(0, image.shape[2]-x))
    len_y = int(np.random.uniform(0, image.shape[3]-y))
    
    noise[:, :, x:x+len_x, y:y+len_y] = torch.normal(0, 1, size=(image.shape[0], image.shape[1], len_x, len_y))

    return image + noise 


def uda_loss(teacher, x_l, y_l, loss_func):
    if np.random.random() > 0.5:
        aug = gaussian_noise(x_l)
        with amp.autocast():
            output, output_sup1, output_sup2 = teacher(aug)
            y_l = y_l.detach()

            loss1 = loss_func(output, y_l)
            loss2 = loss_func(output_sup1, y_l)
            loss3 = loss_func(output_sup2, y_l)
        return loss1 + loss2 + loss3 
    else:
        return 0


def create_meta_pseudo_labels(model, args, images, name, epoch):

    
    # Create directory if not exixts
    if not os.path.exists(args.meta_pseudo_path): #creo cartella metapseudo labels
        os.makedirs(args.Meta_pseudo_path + "/labels")
    if not os.path.exists(args.meta_pseudo_path + "/labels_rgb"): #creo cartella pseudo labels a colori
        os.makedirs(args.Meta_pseudo_path + "/labels_rgb")

    model.eval() 
    model.cuda()

    
    list_of_pred_label = []         
    for nr_img, image in enumerate(images): 
      if image is not None: 
        image = image.unsqueeze(0) 

        output = model(image.cuda()) #predictions from Bisnet
        output = F.softmax(output, dim=1) #Applies a softmax function. It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
        output = F.upsample(output, (512, 1024), mode='nearest')[0] 
        
        output = torch.transpose(output,2,1)
        output = torch.transpose(output,0,2)
        label,prob = torch.argmax(output, axis =2), torch.max(output, axis=2)[0]
        
        
        predicted_label = torch.clone(label)
        predicted_prob = torch.clone(prob)
            
        THRESHOLDS = []
        for i in range(19):
          mask = predicted_label.eq(i) #predicted_prob[predicted_label==i]
          x = torch.masked_select(predicted_prob, mask)

          if len(x) == 0:
            THRESHOLDS.append(0)  

          x = torch.sort(x)[0]
          q = torch.median(x)
          
          THRESHOLDS.append(q.detach().cpu())
          

        THRESHOLDS = np.array(THRESHOLDS)
        THRESHOLDS[THRESHOLDS>0.9]=0.9
        
          
        for i in range(19):
            mask1 = predicted_prob.lt(THRESHOLDS[i])
            mask2 = predicted_label.eq(i)
            mask = torch.logical_and(mask1,mask2)
            predicted_label= predicted_label.masked_fill(mask, 255) 
        
        output = np.array(predicted_label.detach().cpu(), dtype=np.uint8)
        rgb_image = rgb_label(output)
        output = Image.fromarray(output)
        list_of_pred_label.append(predicted_label)
         
      if epoch>48:
        saving_file_name = args.meta_pseudo_path + "/labels/" + name[nr_img] + ".png" 
        saving_file_name_rgb = args.meta_pseudo_path + "/labels_rgb/" + name[nr_img] + ".png" 
        rgb_image.save(saving_file_name_rgb)
        output.save(saving_file_name)
      

    list_of_pred_label = torch.stack(list_of_pred_label)


    return list_of_pred_label


#construct model and optimizer for Generator and Discriminator

def build_model_and_optimizer(args):

    # Build model
    model = BiSeNet(args.num_classes, args.context_path)

    if torch.cuda.is_available() and args.use_gpu:
      model = torch.nn.DataParallel(model).cuda()

    # Build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('Not supported optimizer \n')
        return None
    
    return model, optimizer



def build_model_and_optimizer_discriminator(args):

     if(args.Discriminator==0):
       model_D = FCDiscriminator(num_classes=args.num_classes)
     else: #uso quello light weight 
       model_D= DSCDiscriminator(num_classes=args.num_classes) 
     if torch.cuda.is_available() and args.use_gpu:
       model_D = torch.nn.DataParallel(model_D).cuda()  
     optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rateD, betas=(0.9, 0.99))
     
     return model_D, optimizer_D
 

def build_pretrained_model(args):

  model, opt = build_model_and_optimizer(args)
  model_D, optimizer_D = build_model_and_optimizer_discriminator(args)
  model, model_D, opt, optimizer_D, epoch_start =  upload_model(args, model, model_D, opt, optimizer_D)

  return model, opt, model_D, optimizer_D

# Compute Cross Entropy Loss function 

def compute_loss(model, image, label, loss_func):
    with amp.autocast():
        output, output_sup1, output_sup2 = model(image)
        
        loss1 = loss_func(output, label)
        loss2 = loss_func(output_sup1, label)
        loss3 = loss_func(output_sup2, label)

    return loss1 + loss2 + loss3, output 
