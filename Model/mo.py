#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import cv2
from PIL import *
import pickle
import torch
import torch.utils.data as D
import torchvision 


class GetData(object):
	def __init__(self,path,*args,**k):
		self.path=path
		self.a=args;
		self.k=k;
		self.I=[]
		self.M=[]
	def _read_images(self):
		print(os.listdir(self.path));
		assert('Is.pickle' in os.listdir(self.path) and 'Ms.pickle' in os.listdir(self.path))
		im=os.path.join(self.path,'Is.pickle');
		ms=os.path.join(self.path,'Ms.pickle');
		with open(im,'rb') as f:
			self.I=pickle.load(f)
		with open(ms,'rb') as f:
			self.M=pickle.load(f)
	def __call__(self):
		return (self.I,self.M);




class CreateTrainable(D.Dataset):
	def __init__(self,ims,ms,transform=None):
		self.ims=ims
		self.ms=ms
		self.trans=transform
	def __len__(self):
		assert(len(self.ims))
		return self.ims.__len__()
	def __getitem__(self,indx):
		temp1=self.ims[indx].astype(np.float32)
		temp2=self.ms[indx].astype(np.float32)
		if self.trans:
			temp1=self.trans(temp1)
			temp2=self.trans(temp2)
		return temp1,temp2


def main(path):
	getData=GetData(path)
	getData._read_images();
	H=getData()
	I,M=H
	for i in range(len(I)):
		#only for float32
		I[i]=cv2.resize(I[i].astype(np.float32),(64,64),interpolation=cv2.INTER_AREA);
		M[i]=cv2.resize(M[i].astype(np.float32),(64,64),interpolation=cv2.INTER_AREA);
		I[i]=I[i][:,:,:3]+I[i][:,:,-1].reshape(64,64,1)
	M=list(map(lambda x:x.reshape(64,64,1),M))
	dataset=CreateTrainable(I,M,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.RandomAffine(30)]))
	trainset,valset=D.random_split(dataset,[3300,700])
	trainset=D.DataLoader(dataset=trainset,shuffle=True,batch_size=16);
	valset=D.DataLoader(dataset=valset,shuffle=True,batch_size=16)
	return trainset,valset;

Train,Val=main('../Supporter_/train');
if __name__=="__main__":
	pass
	#Train,Val=main('../Supporter_/train');
	

