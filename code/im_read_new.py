#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 14:47:42 2019

@author: thorsteinngj
"""

import pandas as pd
import numpy as np
import urllib
from tqdm import tqdm


source = '/home/thorsteinngj/Documents/Skoli/Thesis/Code/'
image_csv = pd.read_csv(source+'jewish_dataset.csv')
#%%

img_links = list(image_csv["img_link"])
ids = list(image_csv["grave_id"])
#55

for im in tqdm(range(0,len(image_csv),1)):
    urllib.request.urlretrieve(img_links[im], "/home/thorsteinngj/Documents/Skoli/Thesis/Code/Pictures/Jewish/"+str(ids[im])+".jpg")
