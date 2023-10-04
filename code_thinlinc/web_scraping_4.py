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

def fetch_data(url,header):
    try:
        req = Request(url, headers=header)
        page_raw = urlopen(req).read()
        return page_raw
    except:
        print("Failed trying again") 
        return fetch_data(url,header); 
    

#Start by capturing images of the stones in the training_data.
#source = '/home/thorsteinngj/Documents/Skoli/Thesis/'
#train_data = pd.read_csv(source+'copenhagen_export_graves.csv')
#ids = train_data["graveid"].round(0)


webpage = "https://www.findagrave.com/memorial/"

desktop_eg = pd.read_csv('/zhome/2e/9/124284/Downloads/agents(1).csv')[60:80]
desktop_eg = desktop_eg.values.tolist()

desktop_agents = [item for sublist in desktop_eg for item in sublist]


def random_headers():
    return {'User-Agent': choice(desktop_agents)}
		

ids = random.sample(range(3*50869103+1,4*50869103+1), 300000)

#Url for gravestone
my_dataset =[]

for i in tqdm(range(len(ids))):
	if (i%10000)==0:
		print('Saving file')
		df = pd.DataFrame.from_dict(my_dataset)
		df.to_csv('/zhome/2e/9/124284/environments/gravestones_env/Code/Data/website_data_4.csv')
	else:
		#print("Currently on datapoint :"+str(int(i)))
		
		url_using = webpage + str(int(ids[i]))
		ua = random_headers()
		page_raw = fetch_data(url_using,ua)
        
		#HTML parsing
		page_soup = soup(page_raw, "lxml")
		
		
		picture = page_soup.findAll("meta",{"property":"og:image"})
		
		#Finds all images
		#Linkur a mynd
		image_url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(picture))
		
		#Make sure we get the full sized image, not thumbnail
		for url in image_url:
			big_img_url = url.replace('/photos250','')
			
		if (big_img_url != 'https://www.findagrave.com/assets/images/fg-logo-square.png'):
			

			#Nafn manneskju
			name = page_soup.body.h1
			name_clean = str(name)
			name_cleaned = re.sub("<.*?>", "", name_clean)	
			#Fæðingardot
			#Er allt i page_soup.body.table
			birth_date = page_soup.body.findAll("time",{"id":"birthDateLabel"}) #Birth Date
			bd_clean = link_cleaner(birth_date)
			birth_place = link_cleaner(page_soup.body.findAll("div",{"id":"birthLocationLabel"})) #Birthplace
			
			#Danardot
			death_date = link_cleaner(page_soup.body.findAll("span",{"id": "deathDateLabel"})) #Danardagur
			death_place = link_cleaner(page_soup.body.findAll("div",{"id":"deathLocationLabel"})) #Birthplace
			
			#Cemetary info
			cemetery = link_cleaner(page_soup.body.findAll("span",{"id": "cemeteryNameLabel"})) #Cemetery
			city = link_cleaner(page_soup.body.findAll("span",{"id": "cemeteryCityName"})) #City
			county = link_cleaner(page_soup.body.findAll("span",{"id": "cemeteryCountyName"})) #County
			state = link_cleaner(page_soup.body.findAll("span",{"id": "cemeteryStateName"})) #State
			country = link_cleaner(page_soup.body.findAll("span",{"id": "cemeteryCountryName"})) #County
			
			#Text on profile
			
			#Other images
			
			
			#Texti
			data = {'grave_id': int(ids[i]),'img_link':big_img_url,'name':name_cleaned,"birthday":bd_clean,"birthplace":birth_place,"deathday":death_date,"deathplace":death_place,"cemetary":cemetery,"cem_city":city,"cem_county":county,"cem_state":state,"cem_country":country}
			my_dataset.append(data)
        
        


my_dataset_df = pd.DataFrame.from_dict(my_dataset)
my_dataset_df.to_csv('/zhome/2e/9/124284/environments/gravestones_env/Code/Data/website_data_4.csv')
