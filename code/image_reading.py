#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:20:24 2019

@author: thorsteinngj
"""

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
from random import choice
import random

def link_cleaner(soup_link):
    soup_string = str(soup_link)
    soup_clean = re.sub("<.*?>", "", soup_string)
    return soup_clean

def fetch_data(link,source):
    try:
        urllib.request.urlretrieve(link, source)
    except:
        print("Failed trying again") 
        return fetch_data(link,source)
    

source = '/home/thorsteinngj/Documents/Skoli/Thesis/'
image_csv = pd.read_csv(source+'df_bigger.csv')


#%%

base = '/media/thorsteinngj/c1f0e49d-1218-4d11-ab9f-aadb4a021648/home/thorsteinngj/Documents/Thesis/Pictures/'

#%%

for i in tqdm(range(len(image_csv))):
    if image_csv["cem_country"][i] == 'USA':
        fetch_data(image_csv["img_link"][i],base+'USA/'+image_csv['cem_state'][i]+'/'+str(image_csv['grave_id'][i])+'.jpg')
    else:
        fetch_data(image_csv["img_link"][i],base+image_csv['cem_country'][i]+'/'+str(image_csv['grave_id'][i])+'.jpg')
        

#%%
#55
#Listing up urls and places
urls_us = []
locs_us = []
urls_other = []
locs_other = []

for i in tqdm(range(len(image_csv))): 
    
    if image_csv["cem_country"][i] == 'USA':
        urls_us.append(image_csv["img_link"][i])
        locs_us.append(base+'USA/'+image_csv['cem_state'][i]+'/'+str(image_csv['grave_id'][i])+'.jpg')
    else:
        urls_other.append(image_csv["img_link"][i])
        locs_other.append(base+str(image_csv['cem_country'][i])+'/'+str(image_csv['grave_id'][i])+'.jpg')


#%%
#image_csv["cem_county"]["cem_state"]
#np.unique(image_csv["cem_county"])
        
import urllib.request
import zipfile

#urls = ["http://url.com/archive.zip?key=7UCxcuCzFpYeu7tz18JgGZFAAgXQ2sop", "other url", "another url"]
#filename = "C:/test/test.zip"
#destinationPath = "C:/test"

def url_retriever(urls,locs):
    for i in range(len(urls)):
        try:
            urllib.request.urlretrieve(urls[i],locs[i])
            #sourceZip = zipfile.ZipFile(filename, 'r')
            break
        except ValueError:
            pass

url_retriever(urls_other,locs_other)

#%%
for name in sourceZip.namelist():
    sourceZip.extract(name, destinationPath)
sourceZip.close()
