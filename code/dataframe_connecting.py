#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:05:03 2019

@author: thorsteinngj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from tqdm import tqdm
import seaborn as sn


#Importing dataframes from mine.
df_one = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/df_training.csv')
#df_one = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/df_all_c_fips.csv')


#df_two = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/test_90.csv')
df_two = pd.read_csv('/home/thorsteinngj/Downloads/symbols_g.csv')
df_two = df_two.drop(["Unnamed: 0"],axis=1)
df_two = df_two.drop(["time_elapsed"],axis=1)

df3 = pd.merge(df_one,df_two,on="grave_id")
df5 = df3.drop_duplicates(subset=None, keep='first', inplace=False)

df5["sign"] = 0
df5 = df5.reset_index()
df5 = df5.drop(["index"],axis=1)

for i in range(len(df5)):
    if (df5["no_sign"][i] == 0) and (df5["persons"][i] == 0):
        df5["sign"][i] = 1

df5.fips = df5.fips.astype(int)

df5 = df5[df5.persons == 0]

#%%

#Importing dataframe done by hand

df_g = pd.read_csv('/home/thorsteinngj/Documents/Skoli/Thesis/Dataframes/df_germany_interest.csv')
df_g["sign"] = 0
df_g["no_sign"] = 0

for i in range(len(df_g)):
    if df_g.bible[i] == 1 or df_g.dove[i] == 1 or df_g.praying[i] == 1 or df_g.cross[i] == 1 or df_g.angel[i] == 1 or df_g.fish[i] == 1 or df_g.lamb[i] == 1 or df_g.verbal_relig[i] == 1 or df_g.other_religious[i] == 1:
        df_g.sign[i] = 1
        df_g.no_sign[i] = 0
    else:
        df_g.sign[i] = 0
        df_g.no_sign[i] = 1
    
df_g.fips = df_g.fips.astype(int)



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
        df_new.rel_adh[i] = df_three.rel_adh[df_three.fip == df_new.fips[i]]
    
            
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

df_new = new_dataframe_maker(df5)
df_new_g = new_dataframe_maker(df_g)


#%%

for i in range(len(df5)):
    for j in range(len(df_new)):
        if df5.fips[i] == df_new.fips[j]:
            df5.age[i] = (df5.age[i] - df_new.mean_age[j])/df_new.std_age[j]



df5 = df5[pd.notnull(df5['age'])]
            
#%%

#df5 = df5.reset_index()
groups = np.unique(df_new.pct_group_rel)
groups_g = np.unique(df_new_g.pct_group_rel)
groups_sym = np.unique(df_new.pct_group_sym)

def old_df_group_adding(df5,group,df_new):
    df6 = df5
    df6["group_rel"] = 0
    df6["group_sym"] = 0
    for i in range(len(df6)):
        df6.group_rel[i] = df_new.pct_group_rel[df_new.fips == df5.fips[i]]
    for i in range(len(df6)):
        df6.group_sym[i] = df_new.pct_group_sym[df_new.fips == df5.fips[i]]    
    
    df7 = df6[df6.sign == 1]
    df8 = df6[df6.no_sign == 1]
    
    return df7, df8

df7,df8 = old_df_group_adding(df5,groups,df_new)

df7_g,df8_g = old_df_group_adding(df_g,groups_g,df_new_g)

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
        df_new_2.crosses[i] = sum(df7.crosses[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.bibles[i] = sum(df7.bibles[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.stars[i] = sum(df7.david_star[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.praying[i] = sum(df7.praying_hands[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.angels[i] = sum(df7.angels[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.doves[i] = sum(df7.doves[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.relig[i] = sum(df7.sign[df7.group_rel ==groups[i]])/df_new_2.tot_people[i]
        df_new_2.SEM[i] = df_new_2.std_age[i]/df_new_2.people[i]
        df_new_2.SEM_all[i] = df_new_2.std_age_all[i]/df_new_2.tot_people[i]
        
    return df_new_2

df_relig = new_df_maker(df5,df7)
df_nrelig = new_df_maker(df5,df8)


#%%


labels = df_relig.rel_adh
x_pos = np.arange(len(labels))
CTEs = df_relig.mean_age-df_nrelig.mean_age
error = df_relig.SEM_all

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('Coefficient of Thermal Expansion')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()

#%%
# LATER
ax = df7[df7.group_rel==20].hist(column='age',bins=20,grid=False, figsize=(12,8), color='#86bf91', rwidth=0.9,zorder=2)

def histogram_plotter(df,group,bins,text):
    sn.distplot(df.age[df.group_rel==group],bins=bins,label=text)
    plt.title('Distribution of ages')
    plt.xlabel('Age')
    plt.ylabel('Percentage')
    plt.legend()

histogram_plotter(df8,60,20,'90')
histogram_plotter(df8,70,20,'10')

ax = df7.hist(column='age',by='group_rel',bins=20,grid=False, figsize=(12,8), layout=(5,2), sharex=True, color='#86bf91', rwidth=0.9,zorder=2)
ax.set_xlabel('Age')              
plt.suptitle('Hi')
plt.xlabel('Age')
plt.ylabel('Number of people')
plt.show()
for i,x in enumerate(ax):

    # Despine
    #x.spines['right'].set_visible(False)
    #x.spines['top'].set_visible(False)
    #x.spines['left'].set_visible(False)

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Set x-axis label
    x.set_xlabel("Age reached", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    if i == 1:
        x.set_ylabel("Number of people", labelpad=50, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    x.tick_params(axis='x', rotation=0)
plt.title('A')

df7.hist(column='age',by='group_rel',bins=100)

#%%
df7.hist(column='age',by='group_rel',bins=20,grid=False, figsize=(12,8), layout=(5,2), sharex=True, color='#86bf91', rwidth=0.9,zorder=2)
plt.suptitle('Religiously classified samples',size=20)

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
df7_g.boxplot(column='age',by='group_rel')
plt.title('Age reached for gravestones which have religious symbols, done by hand')
plt.ylabel('Age')
plt.xlabel('Religious adherency percentage')
plt.suptitle('')
plt.show()
df8_g.boxplot(column='age',by='group_rel')
plt.title('Age reached by for gravestones which don\'t have religious symbols, done by hand')
plt.ylabel('Age')
plt.xlabel('Religious adherency percentage')
plt.suptitle('')
plt.show()

#%%
df7.boxplot(column='age',by='group_sym')
plt.title('Age reached for gravestones which have religious symbols')
plt.ylabel('Age')
plt.xlabel('Symbol percentage')
plt.suptitle('')
plt.show()
df8.boxplot(column='age',by='group_sym')
plt.title('Age reached by for gravestones which don\'t have religious symbols')
plt.ylabel('Age')
plt.xlabel('Symbol percentage')
plt.suptitle('')
plt.show()

#%%

df7_g.boxplot(column='age',by='group_sym')
plt.title('Age reached for gravestones which have religious symbols, done by hand')
plt.ylabel('Age')
plt.xlabel('Symbol percentage')
plt.suptitle('')
plt.show()
df8_g.boxplot(column='age',by='group_sym')
plt.title('Age reached by for gravestones which don\'t have religious symbols, done by hand')
plt.ylabel('Age')
plt.xlabel('Symbol percentage')
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
tpr, tnr,ppv,acc = accuracy(praying_mat)
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

angels = np.sum(df_g.angel)
dove = np.sum(df_g.dove)
cross = np.sum(df_g.cross)
bible = np.sum(df_g.bible)
praying = np.sum(df_g.praying)
