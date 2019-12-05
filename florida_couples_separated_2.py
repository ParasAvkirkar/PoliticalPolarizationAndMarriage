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


DATA_PATH = ''
#HISTORY_PATH = "Voter_History_{0}"
PREPROCESSED_PATH = "preprocessed/{0}/florida_processed_{0}.csv"
COUPLES_PATH="couples/{0}/couples_{0}.csv"


# In[ ]:


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--old', dest='old_date', action='store', type=str, help='old date value')
parser.add_argument('--new', dest='new_date', action='store', type=str, help='new date value')

args = parser.parse_args()


# In[3]:


preprocessed_date = args.new_date


# In[4]:


couples_date = args.old_date


# In[47]:


image_file_prefix = couples_date + "_" + preprocessed_date
stat_file_prefix =  couples_date + "_" + preprocessed_date

import os
if not os.path.exists("plots/" + image_file_prefix):
    os.makedirs("plots/" + image_file_prefix)

if not os.path.exists("stats/" + stat_file_prefix):
    os.makedirs("stats/" + stat_file_prefix)
    
stat_file_path = "stats/" + stat_file_prefix + "/"


# In[7]:


global_df = pd.read_csv(PREPROCESSED_PATH.format(preprocessed_date) , sep='\t')


# In[8]:


# global_df.isnull().sum()


# In[7]:


global_df.head()


# In[11]:


req_cols = ['last_name', 'race', 'first_name']
global_df = global_df.dropna(subset=req_cols)


# In[12]:


global_df.isnull().sum()


# In[13]:


global_df.shape


# In[14]:


couples_df = pd.read_csv(COUPLES_PATH.format(couples_date) , sep='\t')


# In[15]:


couples_df[["uniq_addr"]].head()


# ## Collecting Addresses of voters

# In[16]:


addr_df=global_df[['uniq_addr','voter_id']] 


# In[17]:


couples_df_found = pd.merge(couples_df, addr_df, left_on=["voter_id_L"], right_on=["voter_id"], suffixes=["","_L"])


# ### Removing extra column generated of voter id

# In[18]:


del couples_df_found['voter_id']


# In[19]:


couples_df_found = pd.merge(couples_df_found, addr_df, left_on=["voter_id_R"], right_on=["voter_id"], suffixes=["","_R"])


# ### Removing extra column generated of voter id

# In[20]:


del couples_df_found['voter_id']


# In[21]:


couples_df_found[["voter_id_L","voter_id_R", "uniq_addr_L","uniq_addr_R"]].head()


# In[22]:


couples_df_found['separated'] = couples_df_found.apply(lambda x : x.uniq_addr_L != x.uniq_addr_R, axis=1)


# ## Separation percentage

# In[23]:


couples_df_found['separated'].value_counts()


# In[24]:


couples_df_found['separated'].value_counts()/couples_df_found['separated'].shape


# In[25]:


# couples_df_found.isnull().sum()


# In[26]:


couples_df_found.head()


# ## Logging total separation percentage in file

# In[59]:


total_separated_couples = couples_df_found[couples_df_found["separated"] == True].shape[0]
with open(stat_file_path + "total_separation_percentage.csv", "w") as f:
    f.write("\t".join(["separated_count", "total_count", "percentage"]) + "\n")
    f.write("\t".join([str(total_separated_couples), str(couples_df_found.shape[0]), str(100.0 * total_separated_couples/couples_df_found.shape[0])]))


# ## Convert race categorical 

# In[27]:


race_codes = {
    1: "American Indian/Alaskan Native",
    2: "Asian/Pacific Islander",
    3: "Black/Not Hispanic",
    4: "Hispanic",
    5: "White",
    6: "Other",
    7: "Mutli-racial",
    9: "Unknown"
}


# ## Converting race codes to corresponding race-descriptions

# In[28]:


def race_code_lambda(row, subscript):
    if pd.isnull(row["race_" + subscript]):
        return None
    code = int(row["race_" + subscript])
    return race_codes[code]


# ## Creating descriptive race columns is a time consuming process

# In[29]:


couples_df_found["race_desc_L"] = couples_df_found.apply(lambda x: race_code_lambda(x, "L"), axis=1)
couples_df_found["race_desc_R"] = couples_df_found.apply(lambda x: race_code_lambda(x, "R"), axis=1)


# In[30]:


global_df["race_desc"] = global_df["race"].apply(lambda x: race_codes[x])


# ## Feature wise separation percentage

# In[31]:


cols = ["party_affiliation", "gender", "race_desc"]
unique_dic = {}
for c in cols:
    unique_dic[c] = set(global_df[c].unique())


# ## Demographic Percentages

# In[32]:


party_affiliation_counts = {}
race_counts = {}

party_affiliation_percentages = {}
race_percentages = {}

total = global_df.shape[0]
for cat_value in unique_dic["party_affiliation"]:
    party_affiliation_counts[cat_value] = global_df[global_df["party_affiliation"] == cat_value].shape[0]
    party_affiliation_percentages[cat_value] = 100.0 * global_df[global_df["party_affiliation"] == cat_value].shape[0]/total

for cat_value in unique_dic["race_desc"]:
    race_counts[str(cat_value)] = global_df[global_df["race_desc"] == cat_value].shape[0]
    race_percentages[str(cat_value)] =  100.0 * global_df[global_df["race_desc"] == cat_value].shape[0]/total


# ## Thresholding Demographics
# For Race we stick to 5% </br>
# For Party affiliation we stick to 5%

# In[33]:


race_percent_threshold = 5.0
race_other_groups = []
for cat_value in race_percentages:
    if race_percentages[cat_value] <= race_percent_threshold:
        race_other_groups.append(cat_value)


# In[34]:


party_percent_threshold = 5.0
party_other_groups = []
for cat_value in party_affiliation_percentages:
    if party_affiliation_percentages[cat_value] <= party_percent_threshold:
        party_other_groups.append(cat_value)


# # TODO

# In[35]:


# TODO: Implement others mapping into global and couples dataframe
# def create_others_df()


# In[36]:


global_df.shape


# In[37]:


print(str(unique_dic))


# ## Generate Pair-wise stats by feature
# Current focus is only on gender/race/political-affiliation

# In[99]:


import time
from itertools import combinations 

def generate_category_based_on_pair_values(first_val, second_val):
    return "({0}, {1})".format(str(first_val), str(second_val))

def generate_pair_stats_by_feature(global_df, couples_df_found, feature, stats={}, others=[]):
    print("Collecting uniques by feature: " + feature)
    unique_vals = list(global_df[feature].unique())
    cat_combinations = list(combinations(list(unique_vals), 2))
    
    stats[feature] = {}
    print("Processing : " + feature + " : Total comb: " + str(len(cat_combinations)))
    i = 0
    start_time = time.time()
    stats[feature]["Other Categories"] = {'count': 0, 'total': 0}
    for comb in cat_combinations:
        first_val = comb[0]
        second_val = comb[1]
        
        left_right = couples_df_found[(couples_df_found[feature + "_L"] == first_val) & (couples_df_found[feature + "_R"] == second_val)]
        right_left = couples_df_found[(couples_df_found[feature + "_L"] == second_val) & (couples_df_found[feature + "_R"] == first_val)]
        
        count = left_right[left_right["separated"] == True].shape[0]
        count += right_left[right_left["separated"] == True].shape[0]
        
        stat = {}
        stat["count"] = count
        
        if count == 0:
            continue
        
        if first_val in others or second_val in others:
            stats[feature]["Other Categories"]["count"] += count
            stats[feature]["Other Categories"]["total"] += left_right.shape[0] + right_left.shape[0]
            continue
        
        stat["total"] = left_right.shape[0] + right_left.shape[0]
        stat["percent"] = count * 100.0/(left_right.shape[0] + right_left.shape[0])
        
        category = generate_category_based_on_pair_values(first_val, second_val)
        stats[feature][category] = stat
        
        i += 1
        
    print("Total value combinations process: " + str(i) + " Total time (secs): " + str(time.time() - start_time))
    
    print("Processing symmetric combinations: " + feature + " : Total comb: " + str(len(unique_vals)))
    for val in unique_vals:
        subset = couples_df_found[(couples_df_found[feature + "_L"] == val) & (couples_df_found[feature + "_R"] == val)]
        
        count = subset[subset["separated"] == True].shape[0]
        
        if val in others:
            stats[feature]["Other Categories"]["count"] += count
            stats[feature]["Other Categories"]["total"] += subset.shape[0]
            continue
        
        stat = {}
        stat["count"] = count
        
        if count == 0:
            continue
        
        stat["total"] = subset.shape[0]
        stat["percent"] = count * 100.0/(subset.shape[0])
        
        category = generate_category_based_on_pair_values(val, val)
        stats[feature][category] = stat
        
    if len(others) > 0 and stats[feature]["Other Categories"]["count"] > 0:
        stats[feature]["Other Categories"]["percent"] = 100.0 * stats[feature]["Other Categories"]["count"]/stats[feature]["Other Categories"]["total"]
    else:
        del stats[feature]["Other Categories"]
    
    print("Done with processing feature: " + feature)
    
    return stats


# In[100]:


stat_file_path


# In[127]:


def plot_pairwise_stat_by_feature(feature, stats, total_couples=1):
    feature_stats = stats[feature]
    
    figures, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    
    percent_stats = []
    pair_combinations = []
    totals = []
    
    for pair_comb in feature_stats:
        percent_stats.append(feature_stats[pair_comb]["percent"])
        totals.append(feature_stats[pair_comb]["total"])
        pair_combinations.append(pair_comb)

    plot_df = pd.DataFrame({"percent": percent_stats, "category_combination": pair_combinations, "totals": totals})
    plot_df["category_combination_percentage"] = 100.0*plot_df["totals"]/total_couples
    
    sns.barplot(x="percent", y="category_combination", data=plot_df, ax=axes[0], palette=sns.color_palette("Set2"))
    
    axes[0].set(xlabel="Separation percentage")
    
    #     sns.barplot(x="category_combination_percentage", y="category_combination", palette=sns.color_palette("Set2"), data=plot_df, ax=axes[1])

    #   Pie chart
    labels = pair_combinations
    sizes = plot_df["category_combination_percentage"].tolist()
    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['teal', 'salmon', 'silver', 'lightblue', 'orchid', 'pink', 'lightgreen', 'wheat'])
    axes[1].axis('equal')
    axes[1].set(xlabel="Category percentage out of total couples")
    
    
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    
    #     axes[0].set_title(feature)
    #     axes[1].set_title(feature + " wise couple proportion")

    #     Writing stats into file
    stat_file_name = feature + "_totalCouples-" + str(total_couples) + ".csv"
    plot_df.to_csv(stat_file_path + stat_file_name, sep="\t")
        
    plt.savefig("plots/" + image_file_prefix + "/" + image_file_prefix +  "_pairwise_stat_" + feature)
    plt.tight_layout()


# In[123]:


stats = generate_pair_stats_by_feature(global_df, couples_df_found, "race_desc", others=race_other_groups)


# In[128]:


plot_pairwise_stat_by_feature("race_desc", stats, total_couples=couples_df_found.shape[0])


# In[104]:


stats = generate_pair_stats_by_feature(global_df, couples_df_found, "party_affiliation", others=party_other_groups)


# In[129]:


plot_pairwise_stat_by_feature("party_affiliation", stats, total_couples=couples_df_found.shape[0])


# In[106]:


stats = generate_pair_stats_by_feature(global_df, couples_df_found, "gender", others=party_other_groups)


# In[130]:


plot_pairwise_stat_by_feature("gender", stats, total_couples=couples_df_found.shape[0])


# In[108]:


stats


# ## Raw code of combination stats (Obsolete)

# In[109]:


# from itertools import combinations
# total_separated = couples_df_found[couples_df_found["separated"] == True].shape[0]
# stats = {}
# for c in cols:
#     cat_combinations = list(combinations(list(unique_dic[c]), 2))
#     stats[c] = {}
#     print("Processing : " + c + " : Total comb: " + str(len(cat_combinations)))
#     i = 0
#     for comb in cat_combinations:
#         first_val = comb[0]
#         second_val = comb[1]
        
#         left_right = couples_df_found[(couples_df_found[c + "_L"] == first_val) & (couples_df_found[c + "_R"] == second_val)]
#         right_left = couples_df_found[(couples_df_found[c + "_L"] == second_val) & (couples_df_found[c + "_R"] == first_val)]
#         count = left_right[left_right["separated"] == True].shape[0]
#         count += right_left[right_left["separated"] == True].shape[0]
#         stats[c][str(first_val) + "_" + str(second_val)] = count
#         stats[c][str(first_val) + "_" + str(second_val) + "_" + "total"] = left_right.shape[0] + right_left.shape[0]
#         i += 1
        
#     print("Done with comb: " + str(i))
    
#     print("Processing symmetric combinations: " + c + " : Total comb: " + str(len(unique_dic[c])))
#     unique_vals = unique_dic[c]
#     for val in unique_vals:
#         subset = couples_df_found[(couples_df_found[c + "_L"] == val) & (couples_df_found[c + "_R"] == val)]
#         count = subset[subset["separated"] == True].shape[0]
#         stats[c][str(val) + "_" + str(val)] = count
#         stats[c][str(val) + "_" + str(val) + "_" + "total"] = subset.shape[0]
    
#     print("Done with processing column: " + c)


# ## Raw code of plot (Obsolete)

# In[110]:


# figures, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,30))
# race_df = None
# index = 0
# for col in stats:
#     col_stats = stats[col]
#     percent_stats = {}
#     for c in col_stats:
#         if "total" in c:
#             continue
        
#         total = col_stats[c + "_" + "total"]
#         if total == 0:
#             continue
            
#         p = col_stats[c] * 100.0/total
#         if p < 5.0:
#             continue
#         percent_stats[c] = p
    
#     plot_df = pd.DataFrame({"percent": list(percent_stats.values()), "category": list(percent_stats.keys()) })

#     plot_df["percent_str"] = plot_df["percent"].apply(str)
    
# #     sns.barplot(x="category", y="percent", data=plot_df, ax=axes[index])
    
# #     sns.barplot(x="percent_str", y="category", data=plot_df, ax=axes[index])
#     if col == "race":
#         race_df = plot_df
# #         sns.barplot(x="category", y="percent", data=plot_df, ax=axes[index])
#     else:
#         sns.barplot(x="percent", y="category", data=plot_df, ax=axes[index])
#         axes[index].set_title(col)
        

#     index += 1

# plt.tight_layout()


# In[111]:


# sns.barplot(x="percent_round", y="category", data=race_df)


# In[112]:


couples_df_found.age_diff.unique()


# In[113]:


age_diff_total_counts = couples_df_found.groupby(["age_diff"])["age_diff"].agg(["count"]).reset_index()


# In[114]:


age_diff_total_counts.head()


# In[115]:


sns.barplot(x="age_diff", y="count", data=age_diff_total_counts)
plt.savefig("plots/" + image_file_prefix + "/" + image_file_prefix + "_age_diff_aggregation")


# In[116]:


age_diff_separation_stats = couples_df_found[couples_df_found["separated"]==True].groupby(["age_diff"])["age_diff"].agg(["count"]).reset_index()


# In[117]:


age_diff_separation_stats = pd.merge(age_diff_total_counts, age_diff_separation_stats, on=["age_diff"], suffixes=("_total", "_separated"))


# In[118]:


age_diff_separation_stats["percent"] = 100.0 * age_diff_separation_stats["count_separated"]/age_diff_separation_stats["count_total"]


# In[119]:


age_diff_separation_stats


# In[120]:


age_diff_separation_stats.to_csv(stat_file_path + "age_diff_separation_stats.csv", sep='\t', header=["age_diff", "counts_of_that_age_diff", "separated_count", "separated_percentage"])


# In[121]:


sns.barplot(x="age_diff", y="percent", data=age_diff_separation_stats)
plt.savefig("plots/" + image_file_prefix + "/" + image_file_prefix + "_age_diff_separation_percentages")


# In[135]:


import scipy.stats as stats
pearson_corr, corr_pvalue = stats.pearsonr(age_diff_separation_stats['age_diff'], age_diff_separation_stats['percent'])
print('Correlation: {} \nP_value: {}'.format(pearson_corr, corr_pvalue))


# In[ ]:




