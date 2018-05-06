import numpy as np
from logger import *
import torch
from torch.utils.data.dataloader import DataLoader
from my_dataset import My_Dataset
from model import Encoder, Decoder, LAS
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import pdb

def custom_collate(batch):
	batch.sort(key=lambda x: x[0].shape[1], reverse=True)
	data, labels = zip(*batch)
	# make list of transcripts in the batch 
	labels = list(labels)
	# store lengths of each utterance in a list
	seq_lens = [d.shape[1] for d in data]
	# get maximum sequence length of utterances, decides shape of batch 
	min_seq_len = min(seq_lens)
	outputs = torch.zeros(len(batch), 40, min_seq_len).cuda()
	# Update batch dimension for adapting to division in Encoder
	for i, d in enumerate(data):
		outputs[i] = d[:,:min_seq_len]
	return Variable(outputs), labels

def train(save, load, num_epochs, batch_size, learning_rate, validate):
	global vocab
	hidden_size = 256
	#torch.manual_seed(11) # remove this while training
	dev_dataset = My_Dataset(False)
	labels_list = dev_dataset.label_list
	labels_list = set(labels_list)
	labels_list = list(labels_list)
	num_classes = dev_dataset.num_classes
	label_dict = {}
	for i in range(len(labels_list)):
		label_dict[labels_list[i]] = str(i)
	data_loader = DataLoader(dev_dataset, batch_size, shuffle=True, collate_fn=custom_collate)
	model = LAS(num_classes).cuda()
	if(load == True):
		model.load_state_dict(torch.load('model'))
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	
	print(' ---------- Start of Training ---------- ')
	for epoch in range(num_epochs):
		loss = 0
		losses = []
		for idx, (utterances, speaker_ids) in enumerate(data_loader):
			optimizer.zero_grad()
			utterances = Variable(utterances)
			loss = model(utterances, speaker_ids, label_dict)
			pdb.set_trace()
			loss.backward()
			optimizer.step()
			losses.append(loss.data.cpu().numpy())
		print('Epoch: '+str(epoch))
		print('Training Loss: '+ str(np.asscalar(np.mean(losses))))
		if(validate==True):
			validation(batch_size, model)
		if(save==True):
			torch.save(model.state_dict(), 'model1')

def validation(batch_size, model):
	global vocab
	dev_dataset = My_Dataset(True)
	data_loader = DataLoader(dev_dataset, batch_size, shuffle=True, collate_fn=custom_collate)
	loss = 0
	losses = []
	for idx, (utterances, speaker_ids) in enumerate(data_loader):
		loss = model(utterances, speaker_ids)
		losses.append(loss.data.cpu().numpy())
	print('Validation Loss: '+ str(np.asscalar(np.mean(losses))))
	print('')

if __name__ == '__main__':
	num_epochs = 10
	batch_size = 8
	lr = 0.001
	validate = False
	load = False
	save = False
	net = train(save, load, num_epochs, batch_size, lr, validate)