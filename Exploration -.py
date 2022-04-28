#!/usr/bin/env python
# coding: utf-8

# Splitting columns and checking null values

# In[4]:


import pandas as pd
import matplotlib as plt
import numpy as np

ebb1 = pd.read_csv("data/ebb_set1.csv").set_index("customer_id")
ebb2 = pd.read_csv("data/ebb_set2.csv").set_index("customer_id")

set_names = {
    "act": "activations_ebb_set",
    "auto": "auto_refill_ebb_set",
    "deact" : "deactivations_ebb_set",
    #"deprio" : "deprioritizing_ebb_set",
    "inter" : "interactions_ebb_set",
    "ivr" : "ivr_calls_ebb_set",
    "lease" : "lease_history_ebb_set",
    "loyalty" : "loyalty_program_ebb_set",
    "network" : "network_ebb_set",
    "notify" : "notifying_ebb_set",
    "phone" : "phone_data_ebb_set",
    "react" : "reactivations_ebb_set",
    "redempt" : "redemptions_ebb_set",
    "support" : "support_ebb_set",
    "susp" : "suspensions_ebb_set",
    "throt" : "throttling_ebb_set"
}

categorical_features = [
    "manufacturer",
    "operating_system",
    "language_preference",
    "opt_out_email",
    "opt_out_loyalty_email",   
    "opt_out_loyalty_sms",  
    "opt_out_mobiles_ads",
    "opt_out_phone",    
    "marketing_comms_1",  
    "marketing_comms_2",    
    "state",
]

numerical_features = [
    "tenure",
    "number_upgrades",
    "year",
    "total_revenues_bucket",
]

data1 = {
    name: pd.read_csv(f"data/{set_names[name]}1.csv").set_index("customer_id")
        for name in set_names.keys()
}


# In[19]:


act = pd.read_csv("data/network_ebb_set1.csv")
df = ebb1.join(act.set_index('customer_id'))


# Check sets for number of values in columns and see if it needs discarding or filling with averages

# In[35]:


print("Name of column".ljust(30), end='')
print("# rows".ljust(15), end='')
print("# Uniques".ljust(15), end='')
print("% Uniques".ljust(15), end='')
print("# nulls".ljust(15), end='')
print("% NaN".ljust(15), end='')
print()

max_len = 0
max_row = ""

discard_cols = []
null_discard_percent = 50
for name in set_names.keys():
    print("=========== " + str(name) + " ===========")
    for col in data1[name]:
        # Row Name
        print(str(col).ljust(30), end='')
        
        # Row numbers
        rows = len(data1[name][col])
        print(str(rows).ljust(15), end='')
        if max_len < rows:
            max_len = rows
            max_row = col
        
        # Uniques
        uniques = data1[name][col].nunique()
        unique_percent = uniques/rows*100
        print(str(uniques).ljust(15), end='')
        print(str("{:.2f}".format(unique_percent)).ljust(15), end='')
        if uniques == 1 or uniques == 0:
            discard_cols.append(col)
        
        # NAN data
        nulls = data1[name][col].isnull().sum()
        null_percent = nulls/rows*100
        print(str(nulls).ljust(15), end='')
        print(str("{:.2f}".format(null_percent)).ljust(15), end='')
        if null_percent > null_discard_percent:
            discard_cols.append(col)
            
        
        print()
print(str(discard_cols))
print(str(max_row) + " has " + str(max_len) + " rows")


# In[26]:


num_nulls = df.isnull().sum(axis=1)


# In[57]:


# print(num_nulls)
value_counter = {}
for v in num_nulls:
    try:
        value_counter[str(v)] += 1
    except:
        value_counter[str(v)] = 1
print(value_counter)

total = 0
for num in value_counter:
    if int(num) < 8:
        total += value_counter[num]
print(total)


# In[52]:


df3 = df.dropna(thresh=(len(df.columns) - 2))
print(len(df3.index))


# In[ ]:


for name in set_names.keys():
    print(" ===== " + str(name) + " ===== ")
    data = pd.read_csv(f"data/{set_names[name]}1.csv").set_index("customer_id")
    act = pd.read_csv("data/network_ebb_set1.csv").set_index('customer_id')
    print(data.columns)
    print(act.columns)
    fin = data.join(act)
    print(fin.columns)
    print(fin.corr(method='pearson', min_periods=1))


# In[70]:


name = "act"
print(" ===== " + str(name) + " ===== ")
data = pd.read_csv(f"data/{set_names[name]}1.csv").set_index("customer_id")
act = pd.read_csv("data/network_ebb_set1.csv").set_index('customer_id')
print(data.columns)
print(act.columns)
fin = data.join(act)
print(fin.columns)
print(fin.corr(method='pearson', min_periods=1))


# In[169]:


from pathlib import Path  
ebb1 = pd.read_csv("data/ebb_set1.csv").set_index("customer_id")
ebb = ebb1.pop('ebb_eligible')
# display(ebb.dropna())

name = "act"
print(" ===== " + str(name) + " ===== ")
data = pd.read_csv(f"data/{set_names[name]}1.csv").set_index("customer_id")
big_cols = list(data.columns)
converted = pd.DataFrame()
for big_col in big_cols:
    subdata = data[big_col]
    print(big_col)
    uniques = subdata.unique()
    if len(uniques) > 60:
        continue
    else:
        subdata = subdata.to_frame()
        print(uniques)
        for entry in uniques:
            # print(entry)
            kwargs = {str(entry) : (subdata[big_col] == entry).astype(int)}
            subdata = subdata.assign(**kwargs)
        
        # for col in subdata:
        #     print(col)
        #     converted.insert(0, col, subdata[col])
        # print(subdata)
        # convert.insert(0, 


# In[170]:


from pathlib import Path  
ebb1 = pd.read_csv("data/ebb_set1.csv").set_index("customer_id")
ebb = ebb1.pop('ebb_eligible')
# display(ebb.dropna())
# print(type(ebb[0]))

fin = ebb.to_frame().join(subdata)
# print(subdata)
print(fin.columns)
fin = fin.drop("ebb_eligible", axis=1)
print(fin)

fin.corrwith(subdata)


# In[6]:


from pathlib import Path
import pandas as pd
data = pd.read_csv("data/full_ebb_set1.csv").set_index("customer_id")
ebb = data.pop('ebb_eligible')

print("Name of column".ljust(30), end='')
print("# rows".ljust(15), end='')
print("# Uniques".ljust(15), end='')
print("% Uniques".ljust(15), end='')
print("# nulls".ljust(15), end='')
print("% NaN".ljust(15), end='')
print()

max_len = 0
max_row = ""

discard_cols = []
null_discard_percent = 50
for name in data.columns:
    print("=========== " + str(name) + " ===========")
    # Row Name
    print(str(name).ljust(30), end='')

    # Row numbers
    rows = len(data[name])
    print(str(rows).ljust(15), end='')
    if max_len < rows:
        max_len = rows
        max_row = col

    # Uniques
    uniques = data[name].nunique()
    unique_percent = uniques/rows*100
    print(str(uniques).ljust(15), end='')
    print(str("{:.2f}".format(unique_percent)).ljust(15), end='')
    if uniques == 1 or uniques == 0:
        discard_cols.append(name)

    # NAN data
    nulls = data[name].isnull().sum()
    null_percent = nulls/rows*100
    print(str(nulls).ljust(15), end='')
    print(str("{:.2f}".format(null_percent)).ljust(15), end='')
    if null_percent > null_discard_percent:
        discard_cols.append(name)


    print()
print(str(discard_cols))
print(str(max_row) + " has " + str(max_len) + " rows")


# In[16]:


from pathlib import Path
import pandas as pd
data = pd.read_csv("data/full_ebb_set1.csv").set_index("customer_id")
ebb = data.pop('ebb_eligible')


# In[19]:


# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = data, ebb
# summarize class distribution
counter = Counter(y)
X[0, 0]
# print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
    print(label)
    row_ix = where(y == label)[0]
    print(row_ix)
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
# pyplot.legend()
pyplot.show()


# In[7]:


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

ebb1 = pd.read_csv('data/full_ebb_set1.csv').set_index('customer_id')
ebb1_full = ebb1
ebb1_opt = ebb1.iloc[:, 9:14]
columns =  ['total_redemptions', 'tenure',  'number_upgrades','state',  'total_revenues_bucket',  'number_activations', 
            'number_auto_refill',  'number_deactivations', 'number_calls', 'total_quantity',  'voice_minutes', 'total_sms', 
            'total_kb',  'hotspot_kb', 'number_reactivations', 'sum_revenues', 'number_suspensions', 'number_throttles', 'ebb_eligible' ]

ebb1 = ebb1[columns]
print(ebb1_full.shape[0])


# In[2]:


ebb1_full = ebb1_full[ebb1_full['ebb_eligible'] == 1]
print(ebb1_full.shape[0])


# In[3]:


plt.figure(figsize=(60, 6), dpi=80)
sns.countplot(ebb1_full['state'])
plt.show()


# In[4]:


plt.figure(figsize=(6, 6), dpi=80)
sns.countplot(ebb1_full['state'] == "FL")
plt.show()


# In[6]:


#categorical counts
categorical_cols = ['manufacturer', 'operating_system', 'language_preference', 'opt_out_email', 'opt_out_loyalty_email',
       'opt_out_loyalty_sms', 'opt_out_mobiles_ads', 'opt_out_phone', 'state', 'total_revenues_bucket', 'marketing_comms_1', 'marketing_comms_2','act_chan_APP', 'act_chan_BATCH', 'act_chan_BOT', 'act_chan_IVR',
       'act_chan_OTHER', 'act_chan_SMS', 'act_chan_TAS', 'act_chan_WARP',
       'act_chan_WEB']
plt.figure(figsize=(60, 6), dpi=80)
for i in ebb1_full[categorical_cols]:
    sns.countplot(ebb1_full[i])
    plt.show()


# In[ ]:




