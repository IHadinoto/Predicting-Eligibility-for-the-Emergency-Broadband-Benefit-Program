import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
#------------------------------------------------------------------------------
#Preparation
ebb1 = pd.read_csv('data/full_ebb_set1.csv').set_index('customer_id')
ebb1_full = ebb1
ebb1_opt = ebb1.iloc[:, 9:14]
columns =  ['total_redemptions', 'tenure',  'number_upgrades','state',  'total_revenues_bucket',  'number_activations', 
            'number_auto_refill',  'number_deactivations', 'number_calls', 'total_quantity',  'voice_minutes', 'total_sms', 
            'total_kb',  'hotspot_kb', 'number_reactivations', 'sum_revenues', 'number_suspensions', 'number_throttles', 'ebb_eligible' ]

ebb1 = ebb1[columns]


#making binary/categorical variables graphable
print(ebb1_full.columns)
ebb1_full['language_preference'] = ebb1_full['language_preference'].fillna('none')
ebb1_full['opt_out_email'] = ebb1_full['opt_out_email'].fillna(0)
ebb1_full['opt_out_loyalty_email'] = ebb1_full['opt_out_loyalty_email'].fillna(0)
ebb1_full['opt_out_loyalty_sms'] = ebb1_full['opt_out_loyalty_sms'].fillna(0)
ebb1_full['opt_out_mobiles_ads'] = ebb1_full['opt_out_mobiles_ads'].fillna(0)
ebb1_full['opt_out_phone'] = ebb1_full['opt_out_phone'].fillna(0)
ebb1_full['marketing_comms_1'] = ebb1_full['marketing_comms_1'].fillna(0)
ebb1_full['marketing_comms_2'] = ebb1_full['marketing_comms_2'].fillna(0)

# Slight adjustment to "year" column for phone to filter out years above 2022
ebb1_full["year"] = pd.to_numeric(ebb1_full["year"])
over_years = ebb1_full["year"] > 2022
under_years = ebb1_full["year"] <= 2022
avg_year = round(ebb1_full[under_years]["year"].sum()/under_years.sum())
print(avg_year)
print("Years at or under 2022: " + str(under_years.sum()) + "\t Years over 2022: " + str(over_years.sum()))
print(under_years.sum() + over_years.sum() - ebb1_full["year"].shape[0])
ebb1_full.loc[(ebb1_full["year"] > 2022), "year"] = avg_year

# Check for duplicated data
dups = ebb1_full.duplicated()
print(dups.any())

#split data into categorical and numerical
ebb1_full.columns
categorical_cols = ['manufacturer', 'operating_system', 'language_preference', 'opt_out_email', 'opt_out_loyalty_email',
       'opt_out_loyalty_sms', 'opt_out_mobiles_ads', 'opt_out_phone', 'state', 'total_revenues_bucket', 'marketing_comms_1', 'marketing_comms_2','act_chan_APP', 'act_chan_BATCH', 'act_chan_BOT', 'act_chan_IVR',
       'act_chan_OTHER', 'act_chan_SMS', 'act_chan_TAS', 'act_chan_WARP',
       'act_chan_WEB']
numerical_cols = ['last_redemption_date', 'first_activation_date', 'total_redemptions',
       'tenure', 'number_upgrades', 'year', 'number_activations', 'number_auto_refill',
       'number_deactivations', 'number_calls', 'total_quantity',
       'voice_minutes', 'total_sms', 'total_kb', 'hotspot_kb',
       'number_reactivations', 'sum_revenues', 'number_suspensions',
       'number_throttles']

ebb1_numerical = ebb1_full[numerical_cols]
ebb1_categorical = ebb1_full[categorical_cols]

#------------------------------------------------------------------------------
#Graphing

#plot the correlation of all features in a heatmap
plt.figure(figsize=(30,12))
sns.heatmap(ebb1_full.corr().abs(), annot = True)
plt.show()

#Histogram and Boxplot for numerical values
for i in numerical_cols:
    print(i)
    if i in ['last_redemption_date', 'first_activation_date']:
        #only use histogram for dates
        plt.hist(ebb1_full[i])
        plt.show()
        continue
    
    fig, axs = plt.subplots(1,2)
    axs[0].hist(ebb1_full[i])
    axs[1].boxplot(ebb1_full[i])
    plt.show()

#categorical counts of each feature
plt.figure(figsize=(60, 6), dpi=80)
for i in ebb1_full[categorical_cols]:
    sns.countplot(ebb1_full[i])
    plt.show()

#pairplots of numerical cols
sns.pairplot(ebb1_full[numerical_cols])
plt.show()
