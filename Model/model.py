import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as opt
from mo import IS,MS
# Encoder states
# 64X64X3----> conv2d*3 64X64X16 ---> Maxpool --> 32X32X16
# 32X32X16 -----> conv2d*3 32X32X64--->Maxpool ---> 16X16X64
#another output---- > 8*8*128
class DownSampling(nn.Module):
	def __init__(self,shape=3):
		super(DownSampling,self).__init__()
		self.shape=shape
		self.l1=nn.Conv2d(3,16,(3,3),padding=1);
		self.ul1=self.__conv2d(16);
		self.mp=nn.MaxPool2d((2,2));
		self.l2=nn.Conv2d(16,64,(3,3),padding=1);
		self.ul2=self.__conv2d(64);
		self.l3=nn.Conv2d(64,128,(3,3),padding=1)
		self.ul3=self.__conv2d(128);

	def __conv2d(self,dim,bais=False):
		x=nn.BatchNorm2d(dim);
		x=nn.Conv2d(dim,dim,(3,3),padding=1,bias=False);
		x=nn.Conv2d(dim,dim,(3,3),padding=1,bias=False);
		return x;
	def forward(self,x):
		x=self.l1(x)
		x=self.ul1(x)
		x=self.mp(x)
		x=self.l2(x)
		x=self.ul2(x)
		x=self.mp(x)
		x=self.l3(x);
		x=self.ul3(x)
		x=self.mp(x)
		return x;

down=DownSampling();
temp=None;
for a,_ in IS:
	temp=a;
	break;

out=down(temp);
print(out.shape) # 16,128,8,8---> Upsampling
	
