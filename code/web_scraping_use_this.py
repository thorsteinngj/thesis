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

def link_cleaner(soup_link):
    soup_string = str(soup_link)
    soup_clean = re.sub("<.*?>", "", soup_string)
    return soup_clean
    

#Start by capturing images of the stones in the training_data.
source = '/home/thorsteinngj/Documents/Skoli/Thesis/'
train_data = pd.read_csv(source+'copenhagen_export_graves.csv')
ids = train_data["graveid"].round(0)


webpage = "https://www.findagrave.com/memorial/"

desktop_agents = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
             'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
             'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
             'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
             'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
             'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
             'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
             'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
             'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
             'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']
def random_headers():
    return {'User-Agent': choice(desktop_agents)}

#Url for gravestone
my_dataset =[]

for i in tqdm(range(0,6400,1)):
    print("Currently on datapoint :"+str(int(ids[i])))
    
    url_using = webpage + str(int(ids[i]))
    
    ua = random_headers()

    req = Request(url_using, headers=ua)
    page_raw = urlopen(req).read()
    #uClient = urlopen(url_using)
    #page_raw = uClient.read()
    #uClient.close()
    #req.close()


    #HTML parsing
    page_soup = soup(page_raw, "html5lib")
    
    
    picture = page_soup.findAll("meta",{"property":"og:image"})
    
    """#Link to photo page
    xxx = page_soup.findAll("a",{"aria-label":"Photos"})
    photo_site = re.findall(r'<a[^>]* href="([^"]*)"', str(xxx))
    
    photo_url = webpage[0:26]+photo_site[0]
    
    #Open photo image
    
    url_using_pic = photo_url
    uClient = uReq(url_using_pic)
    page_raw_pic = uClient.read()
    uClient.close()
    
    page_soup_pic = soup(page_raw_pic,"html.parser")"""
    
    
    #Finds all images
    #xxx = page_soup.findAll('img')
    #Linkur a mynd
    image_url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(picture))
    
    for url in image_url:
        big_img_url = url.replace('/photos250','')
        
    
    
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
    #print(xxx[2])

    #file1 = open("file"+str(int(ids[i]))+".txt","w")

    #file1.write(str(page_soup))


#Grabs the header of the page 
#Where are the images stored?

#%% Here I create a pandas dataframe out of the dictionary

my_dataset_df = pd.DataFrame.from_dict(my_dataset)


#%%

my_dataset_df.to_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Code/ my_dataset.csv')

#%%


from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError
from bs4 import BeautifulSoup
try:
    html = urlopen(url_using)
except HTTPError as e:
    print(e)
except URLError:
    print("Server down or incorrect domain")
else:
    res = BeautifulSoup(html.read(),"html5lib")
    if res.title is None:
        print("Tag not found")
    else:
        print(res.title)
        
#%%
from selenium import webdriver

browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
browser.get(url_using)
print(browser.find_element_by_class_name("introduction").text)
browser.close()

#%%

for i in range(6265,6267,1):
    print(ids[i])