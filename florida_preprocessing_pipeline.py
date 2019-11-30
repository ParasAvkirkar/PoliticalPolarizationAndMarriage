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


SAVED_PATH_INDIV= "preprocessed/"+date_str+"/florida_processed_"+COUNTY+"_"+date_str+".csv"
SAVED_PATH_AGGR= "preprocessed/"+date_str+"/florida_processed_"+date_str+".csv"


# In[6]:



county_file_name = './'+COUNTY+'_'+date_str+'.txt'


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
 'requested_public_records_exemption',
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
global_df = pd.read_csv( REGISTRATION_PATH.format(date_str) + "/" + county_file_name, sep='\t', names=registration_file_headers, usecols=selective_headers)


print("Read CSV")


# In[11]:


global_df.isnull().sum()


# In[12]:


req_cols = ['residence_city', 'gender', 'birth_date', 'precinct', 'first_name', 'last_name']
global_df = global_df.dropna(subset=req_cols)


# In[13]:


global_df.isnull().sum()


# In[14]:


print(global_df.shape)


# In[16]:


def calculate_age(born):
    born = datetime.strptime(str(born), "%m/%d/%Y").date()
    today = file_date.date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

print("Age calculation")
global_df['age'] = global_df['birth_date'].apply(calculate_age)

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


for col in str_cols_lower:
    global_df[col] = global_df[col].apply(lambda x: str(x).strip().lower())
for col in str_cols_upper:
    global_df[col] = global_df[col].apply(lambda x: str(x).strip().upper())


# In[20]:


global_df['residence_zipcode_5'] = global_df['residence_zipcode'].apply(lambda x: int(str(x)[:5]))


# In[21]:


def generate_zipcode_4(zip):
    zip = str(zip)
    if len(zip) > 5:
        return int(zip[-4:])
    else:
        return ''

global_df['residence_zipcode_4'] = global_df['residence_zipcode'].apply(generate_zipcode_4)


# In[22]:


global_df['uniq_addr'] = global_df[['residence_addr_line_1', 'residence_addr_line_2', 'residence_city', 'residence_zipcode_5']].apply(lambda x: ' '.join([str(y) for y in x]), axis=1)


# In[23]:


import re
global_df['uniq_addr'] = global_df['uniq_addr'].apply(lambda x: re.sub("[^0-9a-zA-Z\s]+", '', x))


# In[24]:


global_df['uniq_addr'] = global_df['uniq_addr'].apply(lambda x: x.strip())

print("Saving file")


def appendDFToCSV(df, csvFilePath, sep="\t"):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


appendDFToCSV(global_df, SAVED_PATH_INDIV)
appendDFToCSV(global_df, SAVED_PATH_AGGR)
