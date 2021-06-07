import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from mo import Train,Val

# Encoder states
# 64X64X3----> conv2d*3 64X64X16 ---> Maxpool --> 32X32X16
# 32X32X16 -----> conv2d*3 32X32X64--->Maxpool ---> 16X16X64
#another output---- > 8*8*128
class DownSampling(nn.Module):
	def __init__(self,shape=3):
		super(DownSampling,self).__init__()
		self.shape=shape
		self.l1=nn.Conv2d(3,32,(3,3),padding=1);
		self.ul1=self.__conv2d(32);
		self.mp=nn.MaxPool2d((2,2));
		self.l2=nn.Conv2d(32,64,(3,3),padding=1);
		self.ul2=self.__conv2d(64);
		self.l3=nn.Conv2d(64,128,(3,3),padding=1)
		self.ul3=self.__conv2d(128);
		self.r=nn.ReLU(inplace=True)

	def __conv2d(self,dim,bias=False):
		x=nn.Sequential(
		       nn.BatchNorm2d(dim),
		       nn.Conv2d(dim,dim,3,bias=bias,padding=1),
		       nn.ReLU(inplace=True),
		       nn.Conv2d(dim,dim,3,bias=bias,padding=1),
		       nn.ReLU(inplace=True)
			)
		return x;
	def forward(self,x):
		L=[]
		x=self.r(self.l1(x))
		x=self.ul1(x)
		L.append(x)
		x=self.mp(x)
		x=self.r(self.l2(x))
		x=self.ul2(x)
		L.append(x)
		x=self.mp(x)
		x=self.r(self.l3(x));
		x=self.ul3(x)
		L.append(x)
		x=self.mp(x)
		return x,L;

#---------------------------------------------------------------------------------------------------------------------------------------------------

class UpSampling(nn.Module):
	def __init__(self,shape=128):
		super(UpSampling,self).__init__()
		self.shape=shape;
		self.l1=self.__transconv2d(128);
		self.l2=self.__conv2d(64);
		self.l3=self.__transconv2d(64);
		self.l4=self.__conv2d(32);
		self.l5=self.__transconv2d(32);
		self.l6=self.__conv2d(16);
		self.l7=nn.Conv2d(16,1,1,bias=False);

	def __transconv2d(self,dim):
		x=nn.Sequential(
		    nn.BatchNorm2d(dim),
		    nn.ConvTranspose2d(dim,dim,4,stride=2,bias=False,padding=1),
		    nn.ReLU(inplace=True)
		 )
		return x;
	def __conv2d(self,dim,bias=False):
		assert(dim%2==0);
		x=nn.Sequential(
		      nn.BatchNorm2d(dim*4),
		      nn.Conv2d(dim*4,dim*2,3,padding=1,bias=True),
		      nn.ReLU(inplace=True),
		      nn.Conv2d(dim*2,dim,3,padding=1,bias=True),
		      nn.ReLU(inplace=True)
		 )
		return x;
	def forward(self,x,L):
		L=L[::-1]
		x=self.l1(x)
		x=torch.cat([x,L[0]],dim=1)
		L.pop(0);
		x=self.l2(x);
		x=self.l3(x);
		x=torch.cat([x,L[0]],dim=1)
		L.pop(0)
		x=self.l4(x)
		x=self.l5(x)
		x=torch.cat([x,L[0]],dim=1)
		L.pop(0)
		x=self.l6(x)
		x=self.l7(x)
		return x;



if __name__=="__main__":
	pass
	# down=DownSampling();
	# upsc=Upsampling();
	# temp=None;
	# tar=None;
	# for a,b in IS:
	# 	temp=a;
	# 	tar=b;
	# 	break;

	# out,L=down(temp);
	# out=upsc(out,L)
	# L=nn.BCEWithLogitsLoss();
	# ff=L(out,tar)
	# print(ff)
