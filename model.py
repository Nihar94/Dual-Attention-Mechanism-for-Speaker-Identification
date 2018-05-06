import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.nn.utils.rnn import *
import torch.optim as optim
import copy
import pdb

class LAS(nn.Module):
	def __init__(self, num_classes):
		super(LAS, self).__init__()
		self.num_classes = num_classes
		self.encoder = Encoder()
		self.decoder = Decoder(self.num_classes)
	def forward(self, utterances, speaker_ids, label_dict):
		attn1, attn1_key, attn2, attn2_key = self.encoder(utterances)
		loss = self.decoder(utterances, speaker_ids, label_dict, self.num_classes, attn1, attn1_key, attn2, attn2_key)
		return loss

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=[5,5], stride=1, padding=2, bias=True)
		self.conv2 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=[5,5], stride=1, padding=2, bias=True)
		self.conv3 = nn.Conv2d(in_channels=15, out_channels=10, kernel_size=[3,3], stride=1, padding=1, bias=True)
		self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=[3,3], stride=1, padding=1, bias=True)
		self.attn2_conv1 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=[3,3], stride=1, padding=1, bias=True)
		self.attn1_conv1 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=[3,3], stride=1, padding=1, bias=True)
		self.attn1_linear1 = nn.Linear(in_features=40, out_features=100)

	def forward(self, utterances):
		utterances = utterances.unsqueeze(1)
		x = self.relu(self.conv1(utterances))
		attn1_x = x
		attn1_x = self.relu(self.attn1_conv1(attn1_x))
		attn1_key = self.relu(self.attn1_conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		attn2_x = x
		attn2_x = self.relu(self.attn2_conv1(attn2_x))
		attn2_key = self.relu(self.attn2_conv1(x))
		return attn1_x, attn1_key, attn2_x, attn2_key


class Decoder(nn.Module):
	def __init__(self, num_classes):
		super(Decoder, self).__init__()
		self.hidden_size = 256
		self.embedding_size = 256
		self.num_classes = num_classes
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=[5,5], stride=1, padding=2, bias=True)
		self.conv2 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=[5,5], stride=1, padding=2, bias=True)
		self.conv3 = nn.Conv2d(in_channels=15, out_channels=1, kernel_size=[3,3], stride=1, padding=1, bias=True)
		self.cl_conv1 = nn.Conv2d(3,6,5)
		self.cl_pool = nn.AvgPool2d(2,2)
		self.cl_adaptive_pool = nn.AdaptiveAvgPool2d((5,500))
		self.cl_conv2 = nn.Conv2d(6,16,5)
		self.cl_conv3 = nn.Conv2d(16,16,5)
		self.cl_fc1 = nn.Linear(16*5*500, self.num_classes)
		self.cl_fc2 = nn.Linear(500, 500)
		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()
		self.loss_fn = nn.CrossEntropyLoss().cuda()
		self.relu = nn.ReLU()
		
	
	def forward(self, utterances, speaker_ids, label_dict, num_classes, attn1_val, attn1_key, attn2_val, attn2_key):
		adaptive_batch_size = utterances.size(0)
		labels = torch.zeros(adaptive_batch_size).cuda().long()
		for i in range(adaptive_batch_size):
			labels[i] = int(label_dict[speaker_ids[i]])
		utterances = utterances.unsqueeze(1)
		# attn1_val = attn1_val.squeeze(1)
		x = self.relu(self.conv1(utterances))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		energy1 = attn1_key*x
		energy2 = attn2_key*x
		attn1 = self.tanh(energy1)
		attn1 = (attn1/torch.sum(attn1))*attn1_val
		attn2 = self.tanh(energy2)
		attn2 = (attn2/torch.sum(attn2))*attn2_val
		x = torch.cat((x, attn1), 1)
		x = torch.cat((x, attn2), 1)
		x = self.cl_pool(self.relu(self.cl_conv1(x)))
		x = self.cl_pool(self.relu(self.cl_conv2(x)))
		# x = self.cl_pool(self.relu(self.cl_conv3(x)))
		# pdb.set_trace()
		x = self.cl_adaptive_pool(x)
		x = x.view(-1, 16*5*500)
		x = self.relu(self.cl_fc1(x))
		
		#for i in range(utterances.size(2)):
		#	utterance_t = x[:,:,:,i]
		#	logits, context = self.forward_step(utterance_t, speaker_ids, attn1, attn1_key, attn2, attn2_key)
			#pdb.set_trace()
		#	loss += self.loss_fn(logits, speaker_ids)
		loss = self.loss_fn(x, labels)
		return loss