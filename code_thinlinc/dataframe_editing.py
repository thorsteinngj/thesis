#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:30:38 2019

@author: s171945
"""

import pandas as pd
import numpy as np
from collections import Counter


from tqdm import tqdm
#from pygal_maps_world.maps import World
df = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Code/website_data_wo_junk.csv')
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


def df_editor(df):
    shape = np.shape(df)[0]
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


def fipser(df,):
    df_fips = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
    shape = np.shape(df)[0]
    shape_fips = np.shape(df_fips)[0]
    for i in tqdm(range(shape)):
        for j in range(shape_fips):
            if (df["cem_state"][i] == df_fips["STNAME"][j]) and (df["cem_county"][i] == df_fips["CTYNAME"][j]):
                df["fips"][i] = df_fips["FIPS"][j]
                
    return df

        

#%%
        
#df = df.drop(["Unnamed: 0","Unnamed: 0.1","Unnamed: 14"],axis=1)
        
merged["birthmonth"] = ""
merged["birthyear"] = ""
merged["fips"] = ""

df_3 = removing_brackets(merged)
df_3 = df_editor(df_3)
df_3 = fipser(df_3)

#%%

df_3.to_csv('/home/thorsteinngj/Documents/Skoli/Thesis/df_bigger.csv')

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
