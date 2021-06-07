#!/usr/bin/env python3 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi']=160;
plt.style.use("fivethirtyeight");
data=pd.read_csv("titan.csv")

print(data.head())
def _prob_class(cat):
	assert(cat in [1,2,3])
	PC=(len(data[data['Pclass']==cat])/len(data))
	PS=(len(data[data["Survived"]==1])/len(data))
	PCGS=(len(data[(data['Survived']==1) & (data['Pclass']==cat)])/len(data))
	return ((PCGS*PC)/PC);

# sns.catplot(x="Pclass",y="Survived",data=data,kind="bar")
# plt.title(f"Probability of survival {_prob_class(1)} for first class, {_prob_class(2)} for second class, {_prob_class(3)} for third class",loc="center",size='10');
# plt.show()
# 

data["Sex"]=data["Sex"].apply(lambda x: 0 if x=="male" else 1)
# fig=plt.figure(figsize=(6,6));
# gs=fig.add_gridspec(2,6); # creates (row,col);
# ax1=fig.add_subplot(gs[0,:-3]);
# ax2=fig.add_subplot(gs[1,:]);
# ax3=fig.add_subplot(gs[0,-3:]);
# # sns.jointplot(ax=ax1,x="Sex",y='Age',kind="kde",hue="Survived",data=data);
# # sns.jointplot(ax=ax2,x="Pclass",y="Age",cmap="jet",kind="kde",hue="Survived",data=data)
# ax1.plot([1,2,3,4,10,100],[1,4,9,16,100,10000])
# ax1.set_title("OK");
# ax1.set_xticks([1,2])
# ax1.set_yticks([2,3])
# ax2.plot([1,3,4,5,6,7],[1,4,8,13,19,26],c="r");
# ax2.set_title("OP");
# ax3.plot([12,3,4,5],[12,5,6,7],c='c');
# ax3.set_title("gg");
# plt.tight_layout()
# plt.show();

# def cmap_plots(cmap_list,ctype):
# 	cmaps=cmap_list
# 	n=cmaps.__len__()
# 	fig=plt.figure(figsize=(8.25,n*0.20),dpi=200)
# 	ax=plt.subplot(1,1,1fra)
# # plt.xticks([0,1]);
Group=data.groupby('Pclass')['Age'].max().unstack()
print(Group);


