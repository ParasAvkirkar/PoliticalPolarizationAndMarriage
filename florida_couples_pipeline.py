#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
import numpy as np
import time
import math
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import datatable as dt

from datetime import datetime
from datetime import date

from xgboost import plot_tree
from os import walk
from os import listdir
from os.path import isfile, join
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import skew
from scipy.special import expit as sigmoid
from scipy.cluster.hierarchy import fclusterdata
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[2]:


DATA_PATH = ''
HISTORY_PATH = "Voter_History_{0}"
REGISTRATION_PATH = "data/{0}_VoterDetail"
COUNTY=sys.argv[1]


# In[3]:


date_str = sys.argv[2]
file_date = datetime.strptime(date_str, "%Y%m%d")

# In[44]:


COUPLES_SAVED_PATH= "couples/"+date_str+"/couples_"+date_str+".csv"
COUPLES_SAVED_PATH_INDIV="couples/"+date_str+"/couples_"+COUNTY+"_"+date_str+".csv"

# In[6]:


def get_county_files():
    registration_path = REGISTRATION_PATH.format(str(date_str))
    county_files = []
    for f in listdir(registration_path):
        if isfile(registration_path + "/" + f):
            county_files.append(registration_path + "/" + f)
    return county_files


# In[7]:


# county_files = get_county_files()


# In[8]:


county_file_name = './'+COUNTY+'_'+date_str+'.txt'
print(county_file_name)

# In[9]:


registration_file_headers = [
    'county_code',
    'voter_id',
    'last_name',
    'suffix',
    'first_name',
    'middle_name',
    'requested_public_records_exemption',
    'residence_addr_line_1',
    'residence_addr_line_2',
    'residence_city',
    'residence_state',
    'residence_zipcode',
    'mail_addr_line_1',
    'mail_addr_line_2',
    'mail_addr_line_3',
    'mail_city',
    'mail_state',
    'mail_zipcode',
    'mail_country',
    'gender',
    'race',
    'birth_date',
    'registration_date',
    'party_affiliation',
    'precinct',
    'precinct_group',
    'precinct_split',
    'precinct_suffix',
    'voter_status',
    'congressional_district',
    'house_district',
    'senate_district',
    'county_commission_district',
    'school_board_district',
    'daytime_area_code',
    'daytime_phone_no',
    'daytime_phone_extension',
    'email_address'
]
selective_headers = [
    'first_name',
    'last_name',
    'birth_date',
    'county_code',
    'residence_addr_line_1',
    'residence_addr_line_2',
    'residence_city',
    'residence_zipcode',
    'gender',
    'race',
    'registration_date',
    'precinct',
    'voter_id',
    'party_affiliation',
    'voter_status'
]


# In[10]:

print("Reading CSV")

global_df = pd.read_csv( REGISTRATION_PATH.format(date_str) + "/" + county_file_name, sep='\t', names=registration_file_headers, usecols=selective_headers)


# In[11]:

print("reading done")
# global_df.isnull().sum()


# In[12]:


req_cols = ['residence_city', 'gender', 'birth_date', 'precinct', 'first_name', 'last_name']
global_df = global_df.dropna(subset=req_cols)


# In[13]:


# global_df.isnull().sum()


# In[14]:


print(global_df.shape)


# In[16]:


def calculate_age(born):
    born = datetime.strptime(str(born), "%m/%d/%Y").date()
    today = file_date.date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

print("Calculating age")
global_df['age'] = global_df['birth_date'].apply(calculate_age)

print("Done calculation age")

# In[18]:


str_cols_lower = [
    'last_name',
    'first_name',
    'residence_addr_line_1',
    'residence_addr_line_2',
    'residence_city'
]
str_cols_upper = [
    'county_code',
    'gender',
    'party_affiliation',
    'voter_status'
]


# In[19]:

print("Handling casing issues")
for col in str_cols_lower:
    global_df[col] = global_df[col].apply(lambda x: str(x).strip().lower())
for col in str_cols_upper:
    global_df[col] = global_df[col].apply(lambda x: str(x).strip().upper())


# In[20]:

print("Handling zipcode 5 issues")
global_df['residence_zipcode_5'] = global_df['residence_zipcode'].apply(lambda x: int(str(x)[:5]))


# In[21]:


def generate_zipcode_4(zip):
    zip = str(zip)
    if len(zip) > 5:
        return int(zip[-4:])
    else:
        return ''

print("Handling zipcode 4 issues")
global_df['residence_zipcode_4'] = global_df['residence_zipcode'].apply(generate_zipcode_4)


# In[22]:


print("Generating unique addresses")
global_df['uniq_addr'] = global_df[['residence_addr_line_1', 'residence_addr_line_2', 'residence_city', 'residence_zipcode_5']].apply(lambda x: ' '.join([str(y) for y in x]), axis=1)


# In[23]:

print("Removing alphanumeric chars from unique addresses")

import re
global_df['uniq_addr'] = global_df['uniq_addr'].apply(lambda x: re.sub("[^0-9a-zA-Z\s]+", '', x))


# In[24]:

print("Removing leading/trailing white-spaces")
global_df['uniq_addr'] = global_df['uniq_addr'].apply(lambda x: x.strip())


# In[26]:


# In[28]:

print("Generating deepcopy: total rows " + str(global_df.shape))
global_df_copy = global_df.copy(deep=True)


# In[29]:

print("Performing self join")
merge = pd.merge(global_df, global_df_copy, on=["uniq_addr"], suffixes=["_L", "_R"])


# In[30]:

print("Considering unequal voter-id combinations")
merge = merge[merge["voter_id_L"] != merge["voter_id_R"]]


# In[31]:

print("Rows after merge: " + str(merge.shape))

# In[32]:


filtered = merge[merge["voter_id_L"] < merge["voter_id_R"]]


# In[33]:


print("Filtering done: rows: " + str(filtered.shape))


# In[34]:


def modified_couple_heuristic(row):
    male_age_threshold = 27
    female_age_threshold = 25
    unknown_age_threshold = 26
    age_diff_threshold = 15
    
    age_diff = abs(row['age_L'] - row['age_R'])
    
    is_age_threshold_L = False
    if row["gender_L"] == "M" and row["age_L"] >= male_age_threshold: 
        is_age_threshold_L = True
    elif row["gender_L"] == "F" and row["age_L"] >= female_age_threshold:
        is_age_threshold_L = True
    elif row["gender_L"] == "U" and row["age_L"] >= unknown_age_threshold:
        is_age_threshold_L = True

    
        
    is_age_threshold_R = False
    if row["gender_R"] == "M" and row["age_R"] >= male_age_threshold: 
        is_age_threshold_R = True
    elif row["gender_R"] == "F" and row["age_R"] >= female_age_threshold:
        is_age_threshold_R = True
    elif row["gender_R"] == "U" and row["age_R"] >= unknown_age_threshold:
        is_age_threshold_R = True
    
    return is_age_threshold_L and is_age_threshold_R and age_diff <= age_diff_threshold
        


# In[55]:

couples = filtered[filtered.apply(modified_couple_heuristic, axis=1)]
print("Couples found: " + str(couples.shape))

# In[38]:


from multiprocessing import  Pool
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# In[39]:


#couples = parallelize_dataframe(filtered, modified_couple_heuristic, 8)


# In[56]:

print('Couples processed so far')

print(couples.shape)


# #### https://datascience.stackexchange.com/questions/26308/after-grouping-to-minimum-value-in-pandas-how-to-display-the-matching-row-resul
# #### https://jamesrledoux.com/code/drop_duplicates

# In[76]:


print("Computing age diff")
couples["age_diff"] = couples.apply(lambda row: abs(row["age_L"] - row["age_R"]), axis=1)


# In[78]:


sorted_couples = couples.sort_values(by="age_diff")


# In[82]:


single_house_couples = sorted_couples.drop_duplicates(subset="uniq_addr", keep="first")
print("Single house couples found: " + str(single_house_couples.shape))

# In[ ]:


def appendDFToCSV(df, csvFilePath, sep="\t"):
    print("appendDFToCSV: " + csvFilePath)
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, index=False, sep=sep, header=False)


# In[ ]:


# appendDFToCSV(single_house_couples, COUPLES_SAVED_PATH)
appendDFToCSV(single_house_couples, COUPLES_SAVED_PATH_INDIV)
