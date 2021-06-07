import os
import numpy as np
import torch.nn as nn
import torch
import torch.optim as opt
import sys
import tqdm
from mo import Train,Val
from model import *

class UNET(nn.Module):
	def __init__(self,up,do):
		super().__init__()
		self.up=up;
		self.do=do
	def forward(self,x):
		g,l=self.do(x)
		x=self.up(g,l)
		return x;


down=DownSampling()
ups=UpSampling()
unet=UNET(ups,down)

epochs=4
loss_f=nn.BCEWithLogitsLoss();
optimizer=opt.Adam(unet.parameters(),lr=3*1e-3);
for i in range(epochs):
	print(f"Epoch {i+1}")
	L=0;
	size=0;
	for a,b in Train:
		size+=a.shape[0];
		optimizer.zero_grad()
		pred=unet(a);
		loss=loss_f(pred,b);
		loss.backward();
		optimizer.step()
		L+=loss.item();
		# print(loss.item())
	if i%2==0:
		print(f"Epoch[{i+1}/14] loss is {L/size}");






