#!/usr/bin/env python
# coding: utf-8

# In[149]:


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


# In[292]:


DATA_PATH = ''
#HISTORY_PATH = "Voter_History_{0}"
PREPROCESSED_PATH = "preprocessed/{0}/florida_processed_{0}.csv"
COUPLES_PATH="couples/{0}/couples_{0}.csv"
#COUPLES_PATH="couples/{0}/couples_FLA_{0}.csv"


# In[293]:


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--old', dest='old_date', action='store', type=str, help='old date value')
parser.add_argument('--new', dest='new_date', action='store', type=str, help='new date value')

args = parser.parse_args()


# In[295]:


preprocessed_date = args.new_date
#preprocessed_date = str(20180313)


# In[296]:


couples_date = args.old_date
#couples_date = str(20140319)


# In[297]:


image_file_prefix = couples_date + "_" + preprocessed_date
stat_file_prefix =  couples_date + "_" + preprocessed_date

import os
if not os.path.exists("plots/" + image_file_prefix):
    os.makedirs("plots/" + image_file_prefix)

if not os.path.exists("stats/" + stat_file_prefix):
    os.makedirs("stats/" + stat_file_prefix)
    
stat_file_path = "stats/" + stat_file_prefix + "/"


# In[298]:


global_df = pd.read_csv(PREPROCESSED_PATH.format(preprocessed_date) , sep='\t')


# In[299]:


# global_df.isnull().sum()


# In[300]:


global_df.head()


# In[301]:


req_cols = ['last_name', 'race', 'first_name']
global_df = global_df.dropna(subset=req_cols)


# In[302]:


global_df.isnull().sum()


# In[303]:


global_df.shape


# In[304]:


couples_df = pd.read_csv(COUPLES_PATH.format(couples_date) , sep='\t')


# In[305]:


couples_df[["uniq_addr"]].head()


# ## Collecting Addresses of voters

# In[306]:


addr_df=global_df[['uniq_addr','voter_id']] 


# In[307]:


couples_df_found = pd.merge(couples_df, addr_df, left_on=["voter_id_L"], right_on=["voter_id"], suffixes=["","_L"])


# ### Removing extra column generated of voter id

# In[308]:


del couples_df_found['voter_id']


# In[309]:


couples_df_found = pd.merge(couples_df_found, addr_df, left_on=["voter_id_R"], right_on=["voter_id"], suffixes=["","_R"])


# ### Removing extra column generated of voter id

# In[310]:


del couples_df_found['voter_id']


# In[311]:


couples_df_found[["voter_id_L","voter_id_R", "uniq_addr_L","uniq_addr_R"]].head()


# In[312]:


couples_df_found['separated'] = couples_df_found.apply(lambda x : x.uniq_addr_L != x.uniq_addr_R, axis=1)


# ## Separation percentage

# In[313]:


couples_df_found['separated'].value_counts()


# In[314]:


couples_df_found['separated'].value_counts()/couples_df_found['separated'].shape


# In[315]:


# couples_df_found.isnull().sum()


# In[316]:


couples_df_found.head()


# ## Logging total separation percentage in file

# In[317]:


total_separated_couples = couples_df_found[couples_df_found["separated"] == True].shape[0]
with open(stat_file_path + "total_separation_percentage.csv", "w") as f:
    f.write("\t".join(["separated_count", "total_count", "percentage"]) + "\n")
    f.write("\t".join([str(total_separated_couples), str(couples_df_found.shape[0]), str(100.0 * total_separated_couples/couples_df_found.shape[0])]))


# ## Convert race categorical 

# In[318]:


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

# In[319]:


def race_code_lambda(row, subscript):
    if pd.isnull(row["race_" + subscript]):
        return None
    code = int(row["race_" + subscript])
    return race_codes[code]


# ## Creating descriptive race columns is a time consuming process

# In[320]:


couples_df_found["race_desc_L"] = couples_df_found.apply(lambda x: race_code_lambda(x, "L"), axis=1)
couples_df_found["race_desc_R"] = couples_df_found.apply(lambda x: race_code_lambda(x, "R"), axis=1)


# In[321]:


global_df["race_desc"] = global_df["race"].apply(lambda x: race_codes[x])


# ## Feature wise separation percentage

# In[322]:


cols = ["party_affiliation", "gender", "race_desc"]
unique_dic = {}
for c in cols:
    unique_dic[c] = set(global_df[c].unique())


# ## Demographic Percentages

# In[323]:


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

# In[324]:


race_percent_threshold = 5.0
race_other_groups = []
for cat_value in race_percentages:
    if race_percentages[cat_value] <= race_percent_threshold:
        race_other_groups.append(cat_value)


# In[325]:


party_percent_threshold = 5.0
party_other_groups = []
for cat_value in party_affiliation_percentages:
    if party_affiliation_percentages[cat_value] <= party_percent_threshold:
        party_other_groups.append(cat_value)


# # TODO

# In[326]:


# TODO: Implement others mapping into global and couples dataframe
# def create_others_df()


# In[327]:


global_df.shape


# In[328]:


print(str(unique_dic))


# ## Generate Pair-wise stats by feature
# Current focus is only on gender/race/political-affiliation

# In[329]:


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


# In[330]:


stat_file_path


# In[331]:


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


# In[332]:


stats = generate_pair_stats_by_feature(global_df, couples_df_found, "race_desc", others=race_other_groups)


# In[333]:


plot_pairwise_stat_by_feature("race_desc", stats, total_couples=couples_df_found.shape[0])


# In[334]:


stats = generate_pair_stats_by_feature(global_df, couples_df_found, "party_affiliation", others=party_other_groups)


# In[335]:


plot_pairwise_stat_by_feature("party_affiliation", stats, total_couples=couples_df_found.shape[0])


# In[336]:


stats = generate_pair_stats_by_feature(global_df, couples_df_found, "gender", others=party_other_groups)


# In[337]:


plot_pairwise_stat_by_feature("gender", stats, total_couples=couples_df_found.shape[0])


# In[338]:


stats


# ## Raw code of combination stats (Obsolete)

# In[339]:


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

# In[340]:


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


# In[341]:


# sns.barplot(x="percent_round", y="category", data=race_df)


# In[342]:


couples_df_found.age_diff.unique()


# In[343]:


age_diff_total_counts = couples_df_found.groupby(["age_diff"])["age_diff"].agg(["count"]).reset_index()


# In[344]:


age_diff_total_counts.head()


# In[345]:


ax = sns.barplot(x="age_diff", y="count", data=age_diff_total_counts)
ax.set(xlabel='age difference between couples', ylabel='couples count')
plt.savefig("plots/" + image_file_prefix + "/" + image_file_prefix + "_age_diff_aggregation")


# In[346]:


age_diff_separation_stats = couples_df_found[couples_df_found["separated"]==True].groupby(["age_diff"])["age_diff"].agg(["count"]).reset_index()


# In[347]:


age_diff_separation_stats = pd.merge(age_diff_total_counts, age_diff_separation_stats, on=["age_diff"], suffixes=("_total", "_separated"))


# In[348]:


age_diff_separation_stats["percent"] = 100.0 * age_diff_separation_stats["count_separated"]/age_diff_separation_stats["count_total"]


# In[349]:


age_diff_separation_stats


# In[350]:


age_diff_separation_stats.to_csv(stat_file_path + "age_diff_separation_stats.csv", sep='\t', header=["age_diff", "counts_of_that_age_diff", "separated_count", "separated_percentage"])


# In[351]:


ax = sns.barplot(x="age_diff", y="percent", data=age_diff_separation_stats, )
ax.set(xlabel='age difference between couples', ylabel='couples separation percent')
plt.savefig("plots/" + image_file_prefix + "/" + image_file_prefix + "_age_diff_separation_percentages")


# In[352]:


# import scipy.stats as stats
# pearson_corr, corr_pvalue = stats.pearsonr(age_diff_separation_stats['age_diff'], age_diff_separation_stats['percent'])
# print('Correlation of Age Diff Vs separation rate: {} \nP_value: {}'.format(pearson_corr, corr_pvalue))


# In[353]:


# cf_polarized = np.where(((couples_df_found["party_affiliation_L"] == "DEM") & (couples_df_found["party_affiliation_R"] == "REP")) |
#                         ((couples_df_found["party_affiliation_L"] == "REP") & (couples_df_found["party_affiliation_R"] == "DEM"))
#                         , 1, 0) 


# In[254]:


# cf_separated = np.where(couples_df_found["separated"] == True, 1, 0)


# In[279]:


#pearson_corr, corr_pvalue = stats.pearsonr(couples_df_stat["no"], couples_df_stat["separation_percent"])
#print('Correlation of Polarized couples (DEM/REP) Vs their separation rate: {} \nP_value: {}'.format(pearson_corr, corr_pvalue))


# In[257]:


# cf_polarized_all = np.where(((couples_df_found["party_affiliation_L"] == "DEM") & (couples_df_found["party_affiliation_R"] == "DEM"))
#                             | ((couples_df_found["party_affiliation_L"] == "REP") & (couples_df_found["party_affiliation_R"] == "REP"))
#                             | ((couples_df_found["party_affiliation_L"] == "NPA") & (couples_df_found["party_affiliation_R"] == "NPA"))
#                         , 0, 1) 


# In[258]:


#pearson_corr, corr_pvalue = stats.pearsonr(cf_polarized_all, cf_separated)
#print('Correlation of Polarized couples Vs their separation rate: {} \nP_value: {}'.format(pearson_corr, corr_pvalue))


# In[262]:


#couples_df_stat = pd.read_csv("stats/20160307_20180313/party_affiliation_totalCouples-2709111.csv" , sep='\t')


# In[274]:


#couples_df_stat.columns = ["no", "separation_percent","cat_comb","total","cat_comb_percent"]


# In[ ]:


#couples_df_stat["no"]


# In[ ]:


from sklearn.metrics import jaccard_score


# In[ ]:


#jaccard_score(cf_polarized_all, cf_separated)  


# In[287]:


#pd.crosstab(cf_polarized_all, cf_separated)


# In[284]:


#from sklearn.metrics import matthews_corrcoef
#matthews_corrcoef(cf_separated, cf_polarized_all)


# In[291]:


from scipy.stats import chi2_contingency
#print(chi2_contingency(pd.crosstab(cf_polarized_all, cf_separated)))

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#cramers_corrected_stat(pd.crosstab(cf_polarized_all, cf_separated))

