#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:55:27 2019

@author: thorsteinngj
"""

import skimage
import numpy as np
import matplotlib.pyplot as plt

img = skimage.io.imread('file:///home/thorsteinngj/Desktop/bible.png')

img_0 = img[:,:,0]/255

filt = np.array([[-0.5,1,-0.5],[-0.5,1,-0.5],[-0.5,1,-0.5]])

dim = np.shape(img_0)

zeros = np.zeros((dim[0]+4,dim[1]+4))


for i in range(dim[0]):
    for j in range(dim[1]):
        zeros[i+2,j+2] = img_0[i][j]
        
        
        
#%% Convolving filter


zeros_2 = np.zeros((dim[0],dim[1]))
        
for i in range(dim[0]):
    for j in range(dim[1]):
        zeros_2[i][j] = np.sum(filt*zeros[i:i+3,j:j+3])
        
        
#%% ReLU
        
relu = np.zeros((dim[0],dim[1]))
        
for i in range(dim[0]):
    for j in range(dim[1]):
        if zeros_2[i][j] < 0:
            relu[i][j] = 0
        else:
            relu[i][j] = zeros_2[i][j]

#%% #max_pool
            
pooled = np.zeros((int(dim[0]/8),int(dim[1]/8)))

for i in range(int(dim[0]/8)):
    for j in range(int(dim[1]/8)):
        pooled[i][j] = np.max(relu[i*8:i*8+8,j*8:j*8+8])

#%%

def plotter(img,text):
    plt.figure()
    plt.imshow(img)
    plt.title(text,fontsize=20)
    plt.tick_params(axis='both',labelsize=0,length=0)
    plt.show()
    
plotter(zeros,'Bible shown with zero-padding')

plotter(zeros_2,'Bible after convolutional layer')

plotter(relu,'Bible after nonlinear layer')

plotter(pooled,'Bible after pooling layer')

plotter(filt,'Filter used')

#%%
zeros_2 = np.zeros((6,6))
con = img_0[:8,:8]*255

for i in range(6):
    for j in range(6):
        zeros_2[i][j] = np.sum(filt*con[i:i+3,j:j+3])
        
relu = np.zeros((6,6))
        
for i in range(6):
    for j in range(6):
        if zeros_2[i][j] < 0:
            relu[i][j] = 0
        else:
            relu[i][j] = zeros_2[i][j]
            
pooled = np.zeros((int(6/2),int(6/2)))

for i in range(int(6/2)):
    for j in range(int(6/2)):
        pooled[i][j] = np.max(relu[i*2:i*2+2,j*2:j*2+2])
            
#%%
            
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, .1)
zero = np.zeros(len(z))
y = np.max([zero, z], axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, y)
ax.set_ylim([-1, 5])
ax.set_xlim([-5, 5])
ax.grid(True)
ax.set_xlabel('x')
ax.set_title('ReLU')

plt.show()
