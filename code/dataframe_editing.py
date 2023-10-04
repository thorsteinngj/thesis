#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:30:38 2019

@author: s171945
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
from tqdm import tqdm
#from pygal_maps_world.maps import World
df = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/copenhagen_export_graves_myformat.csv')
df = df.drop(["Unnamed: 0"],axis=1)
df_2 = pd.read_csv('/home/thorsteinngj/Downloads/website_data.csv')
merged = pd.concat([df,df_2], axis=0,join='inner').sort_index()
merged = merged.reset_index()
merged = merged.drop(["index"],axis=1)

#%%

def removing_brackets(df):
    col_names = []
    for col in df.columns:
        col_names.append(col)
    
    for i in range(len(col_names)):
        print(col_names[i])
        if type(df[col_names[i]][0]) == str:
            df[col_names[i]] = df[col_names[i]].str.replace(r"[\[\]']", '')
    
    return df

def month_fixer(text):
    months = [['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
          ['1','2','3','4','5','6','7','8','9','10','11','12']]
    if text in months[0]:
        #print(months[0].index(text))
        i = months[0].index(text)
        text = text.replace(months[0][i],months[1][i])
        
        return text


def df_editor_birth(df):
    shape = np.shape(df)[0]
    for i in range(shape):
        if len(df["birthday"][i]) == 4:
            df["birthyear"][i] = df["birthday"][i]
            df["birthmonth"][i] = 'Jan'
            df["birthday"][i] ='1'
        elif len(df["birthday"][i]) == 8:
            df["birthmonth"][i] = df["birthday"][i][0:3]
            df["birthyear"][i] = df["birthday"][i][4:]
            df["birthday"][i] = '1'
        elif len(df["birthday"][i]) == 10:
            df["birthmonth"][i] = df["birthday"][i][2:5]
            df["birthyear"][i] = df["birthday"][i][6:]
            df["birthday"][i] = df["birthday"][i][:1]
        elif len(df["birthday"][i]) == 11:
            df["birthmonth"][i] = df["birthday"][i][3:6]
            df["birthyear"][i] = df["birthday"][i][7:]
            df["birthday"][i] = df["birthday"][i][:2]
        else:
            df["birthday"][i] = '31'
            df["birthmonth"][i] = 'Dec'
            df["birthyear"][i] = '10'
    
    return df

def df_editor_death(df):
    shape = np.shape(df)[0]
    for i in range(shape):
        df.deathday[i] = re.split(' \\(',df.deathday[i])[0]

    for i in range(shape):
        if len(df["deathday"][i]) == 4:
            df["deathyear"][i] = df["deathday"][i]
            df["deathmonth"][i] = 'Jan'
            df["deathday"][i] ='1'
        elif len(df["deathday"][i]) == 8:
            df["deathmonth"][i] = df["deathday"][i][0:3]
            df["deathyear"][i] = df["deathday"][i][4:]
            df["deathday"][i] = '1'
        elif len(df["deathday"][i]) == 10:
            df["deathmonth"][i] = df["deathday"][i][2:5]
            df["deathyear"][i] = df["deathday"][i][6:]
            df["deathday"][i] = df["deathday"][i][:1]
        elif len(df["deathday"][i]) == 11:
            df["deathmonth"][i] = df["deathday"][i][3:6]
            df["deathyear"][i] = df["deathday"][i][7:]
            df["deathday"][i] = df["deathday"][i][:2]
        else:
            df["deathday"][i] = '31'
            df["deathmonth"][i] = 'Dec'
            df["deathyear"][i] = '1000'
    
    return df

def df_monthfixer(df):
    shape = np.shape(df)[0]
    for i in range(shape):
        df["deathmonth"][i] = month_fixer(df["deathmonth"][i])
        df["birthmonth"][i] = month_fixer(df["birthmonth"][i])
    return df

def df_agefixer(df):
    df = df.dropna()
    df = df.reset_index()
    df = df.drop("index",axis=1)
    shape = np.shape(df)[0]
    for i in range(shape):
        df["age"][i] = agecalculator(int(df["birthday"][i]),int(df["birthmonth"][i]),int(df["birthyear"][i]),int(df["deathday"][i]),int(df["deathmonth"][i]),int(df["deathyear"][i]))
    return df

def agecalculator(birthday,birthmonth,birthyear,deathday,deathmonth,deathyear):
    age = deathyear-birthyear
    if birthmonth < deathmonth:
        age += (deathmonth-birthmonth)/12
    else:
        #age -= 1
        age += (deathmonth-birthmonth)/12
    if birthday < deathday:
        #Use 30.44 as the average monthday.
        age += (deathday-birthmonth)/365.25
    else:
        #age -= 1/12
        age += (deathday-birthday)/365.25
    return age


def fipser(df,):
    df_fips = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/df_w_fips_val.csv')
    shape = np.shape(df)[0]
    shape_fips = np.shape(df_fips)[0]
    for i in tqdm(range(shape)):
        for j in range(shape_fips):
            if (df["cem_state"][i] == df_fips["STNAME"][j]) and (df["cem_county"][i] == df_fips["CTYNAME"][j]):
                df["fips"][i] = df_fips["FIPS"][j]
                
    return df

        

#%%
        
#df = df.drop(["Unnamed: 0","Unnamed: 0.1","Unnamed: 14"],axis=1)
    

     
def agecalculator(birthday,birthmonth,birthyear,deathday,deathmonth,deathyear):
    age = deathyear-birthyear
    if birthmonth < deathmonth:
        age += (deathmonth-birthmonth)/12
    else:
        #age -= 1
        age += (deathmonth-birthmonth)/12
    if birthday < deathday:
        #Use 30.44 as the average monthday.
        age += (deathday-birthmonth)/365.25
    else:
        #age -= 1/12
        age += (deathday-birthday)/365.25
    return age

text = 'Feb'


    


    
        
df["birthmonth"] = ""
df["birthyear"] = ""
df["deathmonth"] = ""
df["deathyear"] = ""
df["age"] = float(0)
df["fips"] = ""

df = removing_brackets(df)
df = df_editor_birth(df)
df = df_editor_death(df)
df = df_monthfixer(df)
df = df_agefixer(df)
df = fipser(df)

#%%

df.to_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/df_training.csv')

#%%

shape = np.shape(df_new)[0]

def df_editor(df,shape):
    for i in range(shape):
        if len(df["birthday"][i]) == 4:
            df["birthyear"][i] = df["birthday"][i]
            df["birthday"][i] = ""
        elif len(df["birthday"][i]) == 8:
            df["birthmonth"][i] = df["birthday"][i][0:3]
            df["birthyear"][i] = df["birthday"][i][4:]
            df["birthday"][i] = ""
        elif len(df["birthday"][i]) == 10:
            df["birthmonth"][i] = df["birthday"][i][2:5]
            df["birthyear"][i] = df["birthday"][i][6:]
            df["birthday"][i] = df["birthday"][i][:1]
        elif len(df["birthday"][i]) == 11:
            df["birthmonth"][i] = df["birthday"][i][3:6]
            df["birthyear"][i] = df["birthday"][i][7:]
            df["birthday"][i] = df["birthday"][i][:2]
        elif df["birthday"][i] == 'unknown':
            df["birthday"][i] = ""
    
    return df

df_new2 = df_editor(df_new,shape)

#%%

x = Counter(df["cem_state"])

df_fips = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
df_fips.to_csv('/home/thorsteinngj/Documents/Skoli/Thesis/df_fips.csv')
def fipser(df,df_fips):
    shape = np.shape(df)[0]
    shape_fips = np.shape(df_fips)[0]
    for i in tqdm(range(shape)):
        for j in range(shape_fips):
            if (df["cem_state"][i] == df_fips["STNAME"][j]) and (df["cem_county"][i] == df_fips["CTYNAME"][j]):
                df["fips"][i] = df_fips["FIPS"][j]
                
    return df

df_new3 = fipser(df_new2,df_fips)

#%%

#import plotly

import numpy as np
import pandas as pd

df_sample = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
df_sample_r = df_sample[df_sample['STNAME'] == 'California']

values = df_sample_r['TOT_POP'].tolist()
fips = df_sample_r['FIPS'].tolist()

states = Counter(df_new3["fips"])

#values = states.values().tolist()
