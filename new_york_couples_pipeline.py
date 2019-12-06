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


ny_cols = ["last_name", "first_name", "middle_name", "name_suffix", "house_number", "house_fractional_addr", "residence_apartment", "residence_pre_street_direction", "residence_street_name", "residence_post_street_direction", "residence_city", "residence_zip_code_5", "residence_zip_code_4", "mail_addr1", "mail_addr2", "mail_addr3", "mail_addr4", "dob", "gender", "political_party", "other_party", "county_code", "election_district", "legislative_district", "town_city", "ward", "congressional_district", "senate_district", "assembly_district", "last_date_voted", "last_year_voted", "last_county_voted", "last_registered_address", "last_registered_name", "county_voter_registration_no", "application_date", "application_source", "identification_required_flag", "identification_verification_requirement_met_flag", "voter_status_codes", "status_reason_codes", "inactive_voter_date", "purge_voter_date", "unique_nys_voter_id", "voter_history"]


# In[3]:


selective_headers = [
    'first_name',
    'last_name',
    'dob',
    'county_code',
    'house_number',
    'residence_apartment',
    'residence_street_name',
    'residence_city',
    'residence_zip_code_5',
    'gender',
    'unique_nys_voter_id',
    'political_party',
    'voter_status_codes'
]

# precinct and race was not found


# In[4]:


new_york_path = "data/NewYork"


# In[5]:


COUNTY=sys.argv[1]


# In[6]:


date_str = sys.argv[2]


# ## Uncomment below lines and add parameters manually when exploring through notebook instead of python-script

# In[37]:


# date_str = "20121231"
# COUNTY = "31"


# In[9]:


import os
if not os.path.exists(new_york_path + "/couples/" + date_str):
    os.makedirs(new_york_path + "/couples/" + date_str)


# In[10]:


source_county_file_name = "county_" + date_str + "_" + COUNTY + ".csv"


# In[11]:


source_county_file_name


# In[12]:


COUPLES_SAVED_PATH = new_york_path + "/couples/" + date_str + "/" + "couples_" + date_str + "_" + COUNTY + ".csv"


# In[13]:


COUPLES_SAVED_PATH


# In[14]:


global_df = pd.read_csv(new_york_path + "/" + date_str + "_county_files/" + source_county_file_name, sep="\t",  encoding='iso-8859-1')


# In[15]:


global_df.head()


# In[16]:


global_df_copy = global_df.copy(deep=True)


# In[17]:


merge = pd.merge(global_df, global_df_copy, on=["uniq_addr"], suffixes=["_L", "_R"])


# In[18]:


merge = merge[merge["unique_nys_voter_id_L"] != merge["unique_nys_voter_id_R"]]


# In[19]:


merge.shape


# In[20]:


filtered = merge[merge["unique_nys_voter_id_L"] < merge["unique_nys_voter_id_R"]]


# In[21]:


filtered.shape


# In[22]:


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
        


# In[23]:


filtered.columns


# In[24]:


filtered.shape


# In[25]:


couples = filtered[filtered.apply(modified_couple_heuristic, axis=1)]


# In[26]:


couples.shape


# In[27]:


couples.head()


# In[28]:


global_df.shape


# In[29]:


couples["age_diff"] = couples.apply(lambda row: abs(row["age_L"] - row["age_R"]), axis=1)


# In[30]:


sorted_couples = couples.sort_values(by="age_diff")


# In[31]:


single_house_couples = sorted_couples.drop_duplicates(subset="uniq_addr", keep="first")


# In[32]:


single_house_couples.shape


# In[33]:


global_df.shape


# In[34]:


COUPLES_SAVED_PATH


# In[35]:


single_house_couples.to_csv(COUPLES_SAVED_PATH, sep="\t", index=False)


# In[39]:


print("Done processing " + COUPLES_SAVED_PATH)


# In[ ]:




