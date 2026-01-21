# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 11:49:08 2025

@author: tyler
"""

#data visulaization
#import pandas
import pandas as pd
import matplotlib.pyplot as plt
#-------------------------------------
#name the data
data = pd.read_csv("C:/Users/tyler/OneDrive/Documents/BDA/pythonEDAgroup.csv")
data.head()
#-----------------------------------------------------------------------------
#%%
#run chart
fig=plt.figure()
ax1=fig.add_subplot(2,3,1)
ax2=fig.add_subplot(2,3,2)
ax3=fig.add_subplot(2,3,3)
ax4=fig.add_subplot(2,3,4)
ax5=fig.add_subplot(2,3,5)
temp=ax1.plot(data["length"])
temp=ax2.plot(data["width"])
temp=ax3.plot(data["density"])
temp=ax4.plot(data["weight"])
temp=ax5.plot(data["conformity"])
#-----------------------------------------------------------------------------
#%%
#histograms
fig=plt.figure()
ax1=fig.add_subplot(2,3,1)
ax2=fig.add_subplot(2,3,2)
ax3=fig.add_subplot(2,3,3)
ax4=fig.add_subplot(2,3,4)
ax5=fig.add_subplot(2,3,5)
temp=ax1.hist(data["length"])
temp=ax2.hist(data["width"])
temp=ax3.hist(data["density"])
temp=ax4.hist(data["weight"])
temp=ax5.hist(data["conformity"])
#-----------------------------------------------------------------------------
#%%
#scatterplot
fig=plt.figure()
ax1=fig.add_subplot(2,5,1)
ax2=fig.add_subplot(2,5,2)
ax3=fig.add_subplot(2,5,3)
ax4=fig.add_subplot(2,5,4)
ax5=fig.add_subplot(2,5,5)
ax6=fig.add_subplot(2,5,6)
ax7=fig.add_subplot(2,5,7)
ax8=fig.add_subplot(2,5,8)
ax9=fig.add_subplot(2,5,9)
ax10=fig.add_subplot(2,5,10)

temp=ax1.scatter(data["length"],data["width"])
temp=ax2.scatter(data["length"],data["density"])
temp=ax3.scatter(data["length"],data["weight"])
temp=ax4.scatter(data["conformity"],data["length"])
temp=ax5.scatter(data["width"],data["density"])
temp=ax6.scatter(data["width"],data["weight"])
temp=ax7.scatter(data["conformity"],data["weight"])
temp=ax8.scatter(data["conformity"],data["density"])
temp=ax9.scatter(data["weight"],data["density"])
temp=ax10.scatter(data["conformity"],data["weight"])
#-----------------------------------------------------------------------------
#%%
#heatmap for correlation matrix of the bariables
data.head()
correlation=data.corr()
import seaborn as sns
heatmap=sns.heatmap(correlation,annot=True,fmt='.4f')
