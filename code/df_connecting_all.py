#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:05:03 2019

@author: thorsteinngj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sn



# Reading in the dataframes scraped and dataframes from inferenced pictures
df_one = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/df_all_c_fips.csv')

def symbol_reader(loc):
    df = pd.read_csv(loc)
    df = df.drop(["Unnamed: 0"],axis=1)
    df = df.drop(["time_elapsed"],axis=1)
    return df

df_1 = symbol_reader('/home/thorsteinngj/Downloads/symbols_g (1).csv')
df_2 = symbol_reader('/home/thorsteinngj/Downloads/symbols_g_2.csv')
df_3 = symbol_reader('/home/thorsteinngj/Downloads/symbols_g_3.csv')
df_4 = symbol_reader('/home/thorsteinngj/Downloads/symbols_g_4.csv')
df_5 = symbol_reader('/home/thorsteinngj/Downloads/symbols_g_5.csv')

df_sym = pd.concat([df_1,df_2,df_3,df_4,df_5])
#Combining the dataframes
df_com = pd.merge(df_one,df_sym,on="grave_id")
df_com = df_com.drop(["Unnamed: 0"],axis=1)
df_com = df_com.drop(["Unnamed: 0.1"],axis=1)

df_com["sign"] = 0
df_com = df_com.reset_index()
df_com = df_com.drop(["index"],axis=1)
#Classifying them as religious or not
for i in tqdm(range(len(df_com))):
    if (df_com["no_sign"][i] == 0) and (df_com["persons"][i] == 0):
        df_com["sign"][i] = 1
        
#%%
# Splitting the frame into US
df_usa = df_com[df_com.cem_country == 'USA']

fipses = np.unique(df_usa.fips)
df_mis = df_usa[df_usa['fips'].isnull()]
misses = np.unique(df_mis.cem_county)

df_mis = df_mis.reset_index()

def fipser(df,):
    df_fips = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/df_w_fips_val.csv')
    shape = np.shape(df)[0]
    shape_fips = np.shape(df_fips)[0]
    for i in tqdm(range(shape)):
        for j in range(shape_fips):
            if (df["cem_state"][i] == df_fips["STNAME"][j]) and (df["cem_county"][i].lower() == df_fips["CTYNAME"][j].lower()):
                df["fips"][i] = df_fips["FIPS"][j]
                
    return df

df_mis2 = fipser(df_mis)

df_usa2 = pd.concat([df_usa,df_mis2])
df_usa2  = df_usa2.dropna(subset=['fips'])
df_usa2.fips = df_usa2.fips.astype(int)

df_usa2 = df_usa2[df_usa2.persons == 0]

#%%

# Data from other countries
df_row = df_com[df_com.cem_country != 'USA']
df_row = df_row[df_row.persons == 0]
df_row = df_row.reset_index()
df_row = df_row.drop(["index"],axis=1)

for i in range(len(df_row)):
    if (df_row.cem_country[i] == 'England' or df_row.cem_country[i] == 'Scotland' or df_row.cem_country[i] == 'Wales' or df_row.cem_country[i] == 'Northern Ireland'):
        df_row.cem_country[i] = 'United Kingdom'
    elif (df_row.cem_country[i] == 'Caribbean Netherlands' or df_row.cem_country[i] == 'Curaçao'):
        df_row.cem_country[i] = 'abc islands'


#%%

def new_dataframe_maker(df5):
    from collections import Counter
    
    df_three = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/county_reladh_copenhagen.csv')

    
    fips_codes = np.unique(df5.fips)
    numb_fips = Counter(df5.fips)
    
    df_new = pd.DataFrame(np.unique(df5.fips), columns = ['fips'])
    df_new["number_of_people"] = 0
    df_new["mean_age"] = float(0)
    df_new["std_age"] = float(0)
    df_new["number_sign"] = 0
    df_new["median_sign"] = float(0)
    df_new["number_nosign"] = 0
    df_new["median_nosign"] = float(0)
    df_new["pct_symbol"] = float(0)
    df_new["pct_group_rel"] = 0
    df_new["pct_group_sym"] = 0
    df_new["rel_adh"] = float(0)
    
    for i in range(len(fips_codes)):
        df_new.number_of_people[i] = list(numb_fips.values())[i]
        df_new.mean_age[i] = df5.age[df5.fips ==fips_codes[i]].mean()
        df_new.std_age[i] = df5.age[df5.fips ==fips_codes[i]].std()
        df_new.median_sign[i] = df5.age[(df5.fips == fips_codes[i]) & (df5.sign == 1)].median()
        df_new.number_sign[i] = len(df5.age[(df5.fips == fips_codes[i]) & (df5.sign == 1)])
        df_new.median_nosign[i] = df5.age[(df5.fips == fips_codes[i]) & (df5.no_sign == 1)].median()
        df_new.number_nosign[i] = len(df5.age[(df5.fips == fips_codes[i]) & (df5.no_sign == 1)])
        #df_new.rel_adh[i] = df_three.rel_adh[df_three.fip == df_new.fips[i]]
    
            
    for i in range(len(df_new)):
        df_new.number_of_people[i] = df_new.number_sign[i]+df_new.number_nosign[i]
        df_new.pct_symbol[i] = df_new.number_sign[i]/df_new.number_of_people[i]
    
    
    for i in range(len(df_new)):
        for j in np.arange(0,140,10):
            if df_new.rel_adh[i] >= j:
                df_new.pct_group_rel[i] =  int(j)
                
    for i in range(len(df_new)):
        for j in np.arange(0,140,10):
            if df_new.pct_symbol[i] >= j/100:
                df_new.pct_group_sym[i] =  int(j)
                
    
    return df_new

df_new = new_dataframe_maker(df_usa2)
#df_new = df_new[df_new.number_of_people > 99]

df_three = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/county_reladh_copenhagen.csv')

df_new = df_new.reset_index()
df_new = df_new.drop(["index"],axis=1)

for i in range(len(df_new)):
    for j in range(len(df_three)):
        if df_new.fips[i] == df_three.fip[j]:
            df_new.rel_adh[i] = df_three.rel_adh[j]
            
df_new  = df_new.dropna(subset=['rel_adh'])
df_new = df_new[df_new.rel_adh > 0]

df_new = df_new.reset_index()
df_new = df_new.drop(["index"],axis=1)

for i in range(len(df_new)):
        for j in np.arange(0,150,10):
            if df_new.rel_adh[i] >= j:
                df_new.pct_group_rel[i] =  int(j)

#%%
                
df_usa2 = df_usa2.reset_index()
df_usa2 = df_usa2.drop(["index"],axis=1)    
            
groups = np.unique(df_new.pct_group_rel)
groups_sym = np.unique(df_new.pct_group_sym)

df_6 = df_usa2

def old_df_group_adding(df5,group,df_new):
    df6 = df_usa2
    df6["group_rel"] = 0
    df6["group_sym"] = 0
    for i in range(len(df6)):
        try:
            df6.group_rel[i] = df_new.pct_group_rel[df_new.fips == df_usa2.fips[i]]
        except:
            print(i)
    for i in range(len(df6)):
        try:
            df6.group_sym[i] = df_new.pct_group_sym[df_new.fips == df_usa2.fips[i]]
        except:
            print(i)
            
    
    df7 = df6[df6.sign == 1]
    df8 = df6[df6.no_sign == 1]
    
    return df7, df8

df7,df8 = old_df_group_adding(df_usa2,groups,df_new)

#%%
def new_df_maker(df5,df7):
    df_new_2 = pd.DataFrame(groups,columns=['rel_adh'])
    df_new_2["people"] = 0
    df_new_2["tot_people"] = 0
    df_new_2["mean_age"] = float(0)
    df_new_2["std_age"] = float(0)
    df_new_2["std_age_all"] = float(0)
    df_new_2["crosses"] = float(0)
    df_new_2["bibles"] = float(0)
    df_new_2["stars"] = float(0)
    df_new_2["praying"] = float(0)
    df_new_2["angels"] = float(0)
    df_new_2["doves"] = float(0)
    df_new_2["relig"] = float(0)
    df_new_2["SEM"] = float(0)
    df_new_2["SEM_all"] = float(0)
    
    for i in range(len(groups)):
        df_new_2.people[i] = sum(df7.group_rel == groups[i])
        df_new_2.tot_people[i] = sum(df5.group_rel == groups[i])
        df_new_2.mean_age[i] = df7.age[df7.group_rel ==groups[i]].mean()
        df_new_2.std_age[i] = df7.age[df7.group_rel ==groups[i]].std()
        df_new_2.std_age_all[i] = df5.age[df5.group_rel ==groups[i]].std()
        df_new_2.crosses[i] = 100*sum(df7.crosses[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.bibles[i] = 100*sum(df7.bibles[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.stars[i] = 100*sum(df7.david_star[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.praying[i] = 100*sum(df7.praying_hands[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.angels[i] = 100*sum(df7.angels[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.doves[i] = 100*sum(df7.doves[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.relig[i] = 100*sum(df7.sign[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.SEM[i] = df_new_2.std_age[i]/np.sqrt(df_new_2.people[i])
        df_new_2.SEM_all[i] = df_new_2.std_age_all[i]/np.sqrt(df_new_2.tot_people[i])
        
    return df_new_2

df_relig = new_df_maker(df_usa2,df7)
df_nrelig = new_df_maker(df_usa2,df8)



#%%

labels = df_relig.rel_adh
x_pos = np.arange(len(labels))
CTEs = df_relig.mean_age-df_nrelig.mean_age
error = df_relig.SEM_all

fig, ax = plt.subplots()
#plt.rc('font',size=10)
ax.bar(x_pos, CTEs,
       color = 'green',
       yerr=error,
       align='center',
       alpha=0.7,
       ecolor='red',
       capsize=5)
ax.set_ylabel('Mean age reached difference')
ax.set_xlabel('Religious adherency group')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Difference of mean age reached for gravestones classed as religious and non religious')
ax.yaxis.grid(True)

plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
plt.show()

#%%

print(df_relig.corr().loc['mean_age','rel_adh'])
print(df_nrelig.corr().loc['mean_age','rel_adh'])

print(df7.corr().loc['age','group_rel'])
print(df8.corr().loc['age','group_rel'])



#%%
df7.boxplot(column='age',by='group_rel')
plt.title('Age reached for gravestones which have religious symbols')
plt.ylabel('Age')
plt.xlabel('Religious adherency percentage')
plt.suptitle('')
plt.show()
df8.boxplot(column='age',by='group_rel')
plt.title('Age reached for gravestones which don\'t have religious symbols')
plt.ylabel('Age')
plt.xlabel('Religious adherency percentage')
plt.suptitle('')
plt.show()

#%%

def histogram_plotter(df,group,bins,text):
    sn.distplot(df.age[df.group_rel==group],bins=bins,label=text)
    plt.title('Distribution of ages')
    plt.xlabel('Age')
    plt.ylabel('Percentage')
    plt.legend()

histogram_plotter(df7,60,20,'100')
histogram_plotter(df8,60,20,'10')


#%%

df8.hist(column='age',by='group_rel',bins=20,grid=False, figsize=(12,8), layout=(7,2), sharex=True, color='#86bf91', rwidth=0.9,zorder=2)
plt.suptitle('Not religiously classified samples',size=20)

#%%
df7.boxplot(column='age',by='group_sym')
plt.title('Age reached for gravestones which have religious symbols')
plt.ylabel('Age')
plt.xlabel('Symbol percentage')
plt.suptitle('')
plt.show()
df8.boxplot(column='age',by='group_sym')
plt.title('Age reached for gravestones which don\'t have religious symbols')
plt.ylabel('Age')
plt.xlabel('Symbol percentage')
plt.suptitle('')
plt.show()



#%%

def new_dataframe_maker(df5):
    from collections import Counter

    
    fips_codes = np.unique(df5.cem_country)
    numb_fips = Counter(df5.cem_country)
    
    df_new = pd.DataFrame(np.unique(df5.cem_country), columns = ['country'])
    df_new["number_of_people"] = 0
    df_new["mean_age"] = float(0)
    df_new["std_age"] = float(0)
    df_new["number_sign"] = 0
    df_new["median_sign"] = float(0)
    df_new["number_nosign"] = 0
    df_new["median_nosign"] = float(0)
    df_new["pct_symbol"] = float(0)
    df_new["pct_group_rel"] = 0
    df_new["pct_group_sym"] = 0
    df_new["rel_adh"] = float(0)
    
    for i in range(len(fips_codes)):
        df_new.number_of_people[i] = list(numb_fips.values())[i]
        df_new.mean_age[i] = df5.age[df5.cem_country ==fips_codes[i]].mean()
        df_new.std_age[i] = df5.age[df5.cem_country ==fips_codes[i]].std()
        df_new.median_sign[i] = df5.age[(df5.cem_country == fips_codes[i]) & (df5.sign == 1)].median()
        df_new.number_sign[i] = len(df5.age[(df5.cem_country == fips_codes[i]) & (df5.sign == 1)])
        df_new.median_nosign[i] = df5.age[(df5.cem_country == fips_codes[i]) & (df5.no_sign == 1)].median()
        df_new.number_nosign[i] = len(df5.age[(df5.cem_country == fips_codes[i]) & (df5.no_sign == 1)])
        #df_new.rel_adh[i] = df_three.rel_adh[df_three.fip == df_new.fips[i]]
    
            
    for i in range(len(df_new)):
        df_new.number_of_people[i] = df_new.number_sign[i]+df_new.number_nosign[i]
        df_new.pct_symbol[i] = df_new.number_sign[i]/df_new.number_of_people[i]
    
    
    for i in range(len(df_new)):
        for j in np.arange(0,140,10):
            if df_new.rel_adh[i] >= j:
                df_new.pct_group_rel[i] =  int(j)
                
    for i in range(len(df_new)):
        for j in np.arange(0,140,10):
            if df_new.pct_symbol[i] >= j/100:
                df_new.pct_group_sym[i] =  int(j)
                
    
    return df_new

df_row = df_row.dropna(subset=['cem_country'])

df_new_row = new_dataframe_maker(df_row)


df_three = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/country_rasv.csv')

df_new_row = df_new_row.reset_index()
df_new_row = df_new_row.drop(["index"],axis=1)

for i in range(len(df_new_row)):
    for j in range(len(df_three)):
        if df_new_row.country[i] == df_three.country[j]:
            df_new_row.rel_adh[i] = df_three.rel_adh[j]
            
df_new_row  = df_new_row.dropna(subset=['rel_adh'])
df_new_row = df_new_row[df_new_row.rel_adh > 0]

df_new_row = df_new_row.reset_index()
df_new_row = df_new_row.drop(["index"],axis=1)

for i in range(len(df_new_row)):
        for j in np.arange(1.60,3.50,0.01):
            if df_new_row.rel_adh[i] >= j:
                df_new_row.pct_group_rel[i] =  int(j*100)
df_new_row = df_new_row[df_new_row.number_of_people > 499]


 


#%%

df_row = df_row.reset_index()

for i in range(len(df_row)):
    for j in range(len(df_new_row)):
        if df_row.cem_country[i] == df_new_row.country[j]:
            df_row.age[i] = (df_row.age[i] - df_new_row.mean_age[j])/df_new_row.std_age[j]



df5 = df5[pd.notnull(df5['age'])]
            
#%%
groups = np.unique(df_new_row.pct_group_rel)
groups_sym = np.unique(df_new_row.pct_group_sym)

def old_df_group_adding(df5,group,df_new):
    df6 = df5
    df6["group_rel"] = 0
    df6["group_sym"] = 0
    for i in range(len(df6)):
        try:
            df6.group_rel[i] = df_new.pct_group_rel[df_new.country == df5.cem_country[i]]
        except:
            print(i)
    for i in range(len(df6)):
        try:
            df6.group_sym[i] = df_new.pct_group_sym[df_new.country == df5.cem_country[i]]
        except:
            print(i)
    
    df7 = df6[df6.sign == 1]
    df8 = df6[df6.no_sign == 1]
    
    return df7, df8

df7,df8 = old_df_group_adding(df_row,groups,df_new_row)

df7 = df7[df7.group_rel != 0]
df8 = df8[df8.group_rel != 0]
#%%

def new_df_maker(df5,df7):
    df_new_2 = pd.DataFrame(groups,columns=['rel_adh'])
    df_new_2["people"] = 0
    df_new_2["tot_people"] = 0
    df_new_2["mean_age"] = float(0)
    df_new_2["std_age"] = float(0)
    df_new_2["std_age_all"] = float(0)
    df_new_2["crosses"] = float(0)
    df_new_2["bibles"] = float(0)
    df_new_2["stars"] = float(0)
    df_new_2["praying"] = float(0)
    df_new_2["angels"] = float(0)
    df_new_2["doves"] = float(0)
    df_new_2["relig"] = float(0)
    df_new_2["SEM"] = float(0)
    df_new_2["SEM_all"] = float(0)
    
    for i in range(len(groups)):
        df_new_2.people[i] = sum(df7.group_rel == groups[i])
        df_new_2.tot_people[i] = sum(df5.group_rel == groups[i])
        df_new_2.mean_age[i] = df7.age[df7.group_rel ==groups[i]].mean()
        df_new_2.std_age[i] = df7.age[df7.group_rel ==groups[i]].std()
        df_new_2.std_age_all[i] = df5.age[df5.group_rel ==groups[i]].std()
        df_new_2.crosses[i] = 100*sum(df7.crosses[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.bibles[i] = 100*sum(df7.bibles[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.stars[i] = 100*sum(df7.david_star[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.praying[i] = 100*sum(df7.praying_hands[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.angels[i] = 100*sum(df7.angels[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.doves[i] = 100*sum(df7.doves[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.relig[i] = 100*sum(df7.sign[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.SEM[i] = df_new_2.std_age[i]/np.sqrt(df_new_2.people[i])
        df_new_2.SEM_all[i] = df_new_2.std_age_all[i]/np.sqrt(df_new_2.tot_people[i])
        
    return df_new_2

df_relig = new_df_maker(df_row,df7)
df_nrelig = new_df_maker(df_row,df8)

#%%

labels = ['UK - 1.91','France - 1.94','Germany - 2.02','Australia - 2.12','New Zealand - 2.13','Canada - 2.19','Philippines - 3.28']
#labels = df_relig.rel_adh
x_pos = np.arange(len(labels))
CTEs = df_relig.mean_age-df_nrelig.mean_age
error = df_relig.SEM_all

fig, ax = plt.subplots()
#plt.rc('font',size=10)
ax.bar(x_pos, CTEs,
       color = 'green',
       yerr=error,
       align='center',
       alpha=0.7,
       ecolor='red',
       capsize=5)
ax.set_ylabel('Mean age reached difference',size=25)
ax.set_xlabel('Country level religiosity',size=25)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels,size=20,rotation=45)
ax.set_title('Difference of mean age reached for gravestones classed as religious and non religious',size=25)
ax.yaxis.grid(True)
plt.subplots_adjust(bottom=0.35)
#plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
plt.show()

#%%

print(df_relig.corr().loc['mean_age','rel_adh'])
print(df_nrelig.corr().loc['mean_age','rel_adh'])

print(df7.corr().loc['age','group_rel'])
print(df8.corr().loc['age','group_rel'])


#%%
df8.hist(column='age',by='group_rel',bins=100)

df7.hist(column='age',by='group_rel',bins=100)

#%%

df8.hist(column='age',by='cem_country',bins=20,grid=False, figsize=(12,8), layout=(4,2), sharex=True, color='#86bf91', rwidth=0.9,zorder=2)
plt.suptitle('Non-religiously classified samples',size=20)


#%%

locs = np.arange(1,8)

df7.boxplot(column='age',by='group_rel')
plt.title('Age reached for gravestones which have religious symbols',size=25)
plt.ylabel('Age',size=25)
plt.xticks(locs,labels,size=20,rotation=45)
plt.xlabel('Country level religiosity',size=25)
plt.subplots_adjust(bottom=0.35)
plt.suptitle('')
plt.show()
df8.boxplot(column='age',by='group_rel')
plt.title('Age reached for gravestones which don\'t have religious symbols',size=25)
plt.ylabel('Age',size=25)
plt.xticks(locs,labels,size=20,rotation=45)
plt.xlabel('Country level religiosity',size=25)
plt.subplots_adjust(bottom=0.35)
plt.suptitle('')
plt.show()

#%%
locs = np.arange(1,8)

df7.boxplot(column='age',by='group_rel')
plt.title('Age reached for gravestones which have religious symbols')
plt.ylabel('Age')
plt.xticks(locs,labels,rotation=45)
plt.xlabel('Country level religiosity')
plt.subplots_adjust(bottom=0.35)
plt.suptitle('')
plt.show()
df8.boxplot(column='age',by='group_rel')
plt.title('Age reached for gravestones which don\'t have religious symbols')
plt.ylabel('Age')
plt.xticks(locs,labels,rotation=45)
plt.xlabel('Country level religiosity')
plt.subplots_adjust(bottom=0.35)
plt.suptitle('')
plt.show()

#%%
# library
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
 
# Get the data (csv file is hosted on the web)
url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
data = pd.read_csv(url)
 
# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]
 
# And transform the old column name in something numeric
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes
 
# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df7['group_rel'], df7['group_sym'], df7['age'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()
 
# to Add a color bar which maps values to colors.
ax.plot_trisurf(df7['group_rel'], df7['group_sym'], df7['age'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( plt.surf, shrink=0.5, aspect=5)
plt.show()
 
# Rotate it
ax.view_init(30, 45)
plt.show()
 
# Other palette
ax.plot_trisurf(df7['group'], df7['group_sym'], df7['age'], cmap=plt.cm.jet, linewidth=0.2)
plt.show()


#%%

df_test = df5[df5.persons == 0]
df_g_test = df_g[df_g.grave_id.isin(df_test.grave_id)]
df_g_test = df_g_test.drop_duplicates(subset='grave_id', keep='first', inplace=False)

df_combine = pd.DataFrame(np.unique(df_test.grave_id), columns = ['grave_id'])
df_combine["bible"] = 0
df_combine["bible_g"] = 0
df_combine["cross"] = 0
df_combine["cross_g"] = 0
df_combine["dove"] = 0
df_combine["dove_g"] = 0
df_combine["praying"] = 0
df_combine["praying_g"] = 0
df_combine["angel"] = 0
df_combine["angel_g"] = 0
df_combine["no_sign"] = 0
df_combine["no_sign_g"] = 0
df_combine["sign"] = 0
df_combine["sign_g"] = 0

for i in range(len(df_combine)):
    df_combine.bible[i] = df_test.bibles[df_combine.grave_id[i]==df_test.grave_id]
    df_combine.cross[i] = df_test.crosses[df_combine.grave_id[i]==df_test.grave_id]
    df_combine.dove[i] = df_test.doves[df_combine.grave_id[i]==df_test.grave_id]
    df_combine.praying[i] = df_test.praying_hands[df_combine.grave_id[i]==df_test.grave_id]
    df_combine.angel[i] = df_test.angels[df_combine.grave_id[i]==df_test.grave_id]
    df_combine.no_sign[i] = df_test.no_sign[df_combine.grave_id[i]==df_test.grave_id]
    df_combine.sign[i] = df_test.sign[df_combine.grave_id[i]==df_test.grave_id]
    df_combine.bible_g[i] = df_g_test.bible[df_combine.grave_id[i]==df_g_test.grave_id]
    df_combine.cross_g[i] = df_g_test.cross[df_combine.grave_id[i]==df_g_test.grave_id]
    df_combine.dove_g[i] = df_g_test.dove[df_combine.grave_id[i]==df_g_test.grave_id]
    df_combine.praying_g[i] = df_g_test.praying[df_combine.grave_id[i]==df_g_test.grave_id]
    df_combine.angel_g[i] = df_g_test.angel[df_combine.grave_id[i]==df_g_test.grave_id]
    df_combine.no_sign_g[i] = df_g_test.no_sign[df_combine.grave_id[i]==df_g_test.grave_id]
    df_combine.sign_g[i] = df_g_test.sign[df_combine.grave_id[i]==df_g_test.grave_id]
    
    
for i in range(len(df_combine)):
    if df_combine.bible[i] != 0:
        df_combine.bible[i] = 1
    if df_combine.cross[i] != 0:
        df_combine.cross[i] = 1
    if df_combine.dove[i] != 0:
        df_combine.dove[i] = 1
    if df_combine.praying[i] != 0:
        df_combine.praying[i] = 1
    if df_combine.angel[i] != 0:
        df_combine.angel[i] = 1
        
        
#%%
    
# Now the confusion matrix
        
crosses_tp = 0
crosses_fp = 0
crosses_fn = 0
crosses_tn = 0

def confusion_matrix(df,label_1,label_2):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(df)):
        if df_combine[label_1][i] == 1 and df_combine[label_2][i] == 1:
            tp +=1
        elif df_combine[label_1][i] == 1 and df_combine[label_2][i] == 0:
            fp +=1
        elif df_combine[label_1][i] == 0 and df_combine[label_2][i] == 1:
            fn +=1
        elif df_combine[label_1][i] == 0 and df_combine[label_2][i] == 0:
            tn +=1
            
    conf_mat = [[tp,fp],[fn,tn]]
    return np.array(conf_mat)

cross_mat = confusion_matrix(df_combine,'cross','cross_g')
angel_mat = confusion_matrix(df_combine,'angel','angel_g')
bible_mat = confusion_matrix(df_combine,'bible','bible_g')
dove_mat = confusion_matrix(df_combine,'dove','dove_g')
praying_mat = confusion_matrix(df_combine,'praying','praying_g')
religion_mat = confusion_matrix(df_combine,'sign','sign_g')

#%%

# I want to plot the confusion matrices

#%%

def accuracy(mat):
    tpr = mat[0][0]/(mat[0][0]+ mat[1][0])
    tnr = mat[1][1]/(mat[1][1]+mat[0][1])
    ppv = mat[0][0]/(mat[0][0]+mat[0][1])
    acc = (mat[0][0]+mat[1][1])/(mat[0][0]+mat[1][1]+mat[0][1]+mat[1][0])
    
    return tpr, tnr,ppv,acc

print(praying_mat)
tpr, tnr,ppv,acc = accuracy(religion_mat)
print("Sensitivity is: "+str(tpr))
print("Specificity is: "+str(tnr))
print("Precision is: "+str(ppv))
print("Accuracy is: "+str(acc))
    
#%%
#Confusion matrix plot with heat map.
import seaborn as sns
plt.rcParams['figure.figsize'] = 8,6
#cm_nm0000168 = confusion_matrix(y_test_nm0000168, y_pred_nm0000168)
#df_cm_nm0000168 = pd.DataFrame(cross_mat, range(2), range(2))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
# Show confusion matrix in a separate window
#plt.matshow(cm)
ax = sn.heatmap(religion_mat, annot=True,annot_kws={"size": 16},cmap=plt.cm.Blues, fmt='g')# font size
ax.tick_params(axis='both', which='both', length=0,labelsize=0)
plt.title('Confusion matrix for all')
#plt.colorbar()
plt.ylabel('Predicted label')
plt.xlabel('Actual label')
plt.show()            
    
#%%

flights = sns.load_dataset("flights")
