import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import pdb

class My_Dataset(Dataset):
	def __init__(self, val):
		self.dir = '/media/nihar/My Passport/train_mels/'
		self.f_list = []
		self.label_list = []
		for f in os.listdir(self.dir):
			if os.path.isfile(os.path.join(self.dir,f)):
				self.f_list.append(f)
		for f in os.listdir(self.dir):
			if os.path.isfile(os.path.join(self.dir,f)):
				self.label_list.append(f[0:f.find('-')])
		self.length = len(self.f_list)
		self.num_classes = len(set(self.label_list))

	def __getitem__(self,index):
		data_index = self.f_list[index]
		ret_data = torch.from_numpy(np.load(self.dir+self.f_list[index])).cuda()
		ret_labels = self.f_list[index][0:self.f_list[index].find('-')]
		return Variable(ret_data), ret_labels

	def __len__(self):
		return self.length