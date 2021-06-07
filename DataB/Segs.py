#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sys
from threading import *
from multiprocessing import *
import multiprocessing as mp
import matplotlib.pyplot as plt
from PIL import Image
import pickle
plt.rcParams["figure.dpi"]=140;
plt.style.use("fivethirtyeight")
# print(os.listdir())
train_csv=pd.read_csv('../Supporter_/train.csv');
Nulls=train_csv.isna().sum().sort_values(ascending=False)
# print(Nulls)
path='../Supporter_/train'
print(os.listdir(path))
signal_=False

def cats_(cat,pat,Is,m=1e20):
	for k in os.listdir(cat):
		U=np.array(Image.open(os.path.join(pat,k)))
		Is.append(U)
		m=max(m,U.shape[0]*U.shape[1]);
	return m

I=[]
M=[]
if 'Is.pickle' and 'Ms.pickle' in os.listdir(path):
	with open('../Supporter_/train/Is.pickle','rb') as f:
		I=pickle.load(f)
	with open('../Supporter_/train/Ms.pickle','rb') as f:
		M=pickle.load(f)
	print("Data Loaded from previous exec")
else:
	signal_=True
	try:
		t1=Thread(target=cats_,args=('../Supporter_/images',os.path.join(path,'images'),I,))
		t2=Thread(target=cats_,args=('../Supporter_/masks',os.path.join(path,'masks'),M,))
		t1.start()
		t2.start()
		t1.join()
		t2.join()
	except:
		raise('Some error occurred during reading the images and masks')
	with open("../Supporter_/train/Is.pickle",'wb') as pkl:
		pickle.dump(I,pkl)
	with open("../Supporter_/train/Ms.pickle","wb") as pkl:
		pickle.dump(M,pkl)
	print('Data is saved in corresponding files')



#clipping pixel intensities of mask [0..1]
def clip_(image,indx,List):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i][j]<=0.0000015:
				image[i][j]=0.
			else:
				image[i][j]=1.
	List[indx]=image

	return image


if signal_:
	Ts=[]
	for i in range(len(M)):
		t=Thread(target=clip_,args=(M[i],i,M,))
		t.start()
		Ts.append(t)
	for k in Ts:
		k.join()
	with open('../Supporter_/train/Ms.pickle','wb') as f:
		pickle.dump(M,f)

if __name__=="__main__":
	fig=plt.figure(figsize=(12,12))
	gs=plt.GridSpec(2,3)
	ax1=fig.add_subplot(gs[0,:]);
	ax2=fig.add_subplot(gs[1,:-1]);
	ax1.set_title("Image ");
	ax2.set_title("Mask");
	print(I[-10].shape,M[-10].shape)
	ax1.imshow(I[-10],cmap=plt.cm.jet);
	ax2.imshow(M[-10]);
	plt.show()