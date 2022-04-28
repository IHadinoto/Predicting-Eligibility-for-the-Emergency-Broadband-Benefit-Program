#--------------------------------------------------------------------------------
#Exploratory Data Analysis
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

ebb1 = pd.read_csv("data/ebb_set1.csv").set_index("customer_id")
ebb2 = pd.read_csv("data/ebb_set2.csv").set_index("customer_id")
ebb_eval = pd.read_csv("data/eval_set.csv").set_index("customer_id")

### Reading data manually
act1 = pd.read_csv("data/activations_ebb_set1.csv")
auto1 = pd.read_csv("data/auto_refill_ebb_set1.csv")
deact1 = pd.read_csv("data/deactivations_ebb_set1.csv")
inter1 = pd.read_csv("data/interactions_ebb_set1.csv")
ivr1 = pd.read_csv("data/ivr_calls_ebb_set1.csv")
lease1 = pd.read_csv("data/lease_history_ebb_set1.csv")
loyalty1 = pd.read_csv("data/loyalty_program_ebb_set1.csv")
network1 = pd.read_csv("data/network_ebb_set1.csv")
notify1 = pd.read_csv("data/notifying_ebb_set1.csv")
phone1 = pd.read_csv("data/phone_data_ebb_set1.csv")
react1 = pd.read_csv("data/reactivations_ebb_set1.csv")
redempt1 = pd.read_csv("data/redemptions_ebb_set1.csv")
support1 = pd.read_csv("data/support_ebb_set1.csv")
susp1 = pd.read_csv("data/suspensions_ebb_set1.csv")
throt1 = pd.read_csv("data/throttling_ebb_set1.csv")

act2 = pd.read_csv("data/activations_ebb_set2.csv")
auto2 = pd.read_csv("data/auto_refill_ebb_set2.csv")
deact2 = pd.read_csv("data/deactivations_ebb_set2.csv")
inter2 = pd.read_csv("data/interactions_ebb_set2.csv")
ivr2 = pd.read_csv("data/ivr_calls_ebb_set2.csv")
lease2 = pd.read_csv("data/lease_history_ebb_set2.csv")
loyalty2 = pd.read_csv("data/loyalty_program_ebb_set2.csv")
network2 = pd.read_csv("data/network_ebb_set2.csv")
notify2 = pd.read_csv("data/notifying_ebb_set2.csv")
phone2 = pd.read_csv("data/phone_data_ebb_set2.csv")
react2 = pd.read_csv("data/reactivations_ebb_set2.csv")
redempt2 = pd.read_csv("data/redemptions_ebb_set2.csv")
support2 = pd.read_csv("data/support_ebb_set2.csv")
susp2 = pd.read_csv("data/suspensions_ebb_set2.csv")
throt2 = pd.read_csv("data/throttling_ebb_set2.csv")

act_eval = pd.read_csv("data/activations_eval_set.csv")
auto_eval = pd.read_csv("data/auto_refill_eval_set.csv")
deact_eval = pd.read_csv("data/deactivations_eval_set.csv")
inter_eval = pd.read_csv("data/interactions_eval_set.csv")
ivr_eval = pd.read_csv("data/ivr_calls_eval_set.csv")
lease_eval = pd.read_csv("data/lease_history_eval_set.csv")
loyalty_eval = pd.read_csv("data/loyalty_program_eval_set.csv")
network_eval = pd.read_csv("data/network_eval_set.csv")
notify_eval = pd.read_csv("data/notifying_eval_set.csv")
phone_eval = pd.read_csv("data/phone_data_eval_set.csv")
react_eval = pd.read_csv("data/reactivations_eval_set.csv")
redempt_eval = pd.read_csv("data/redemptions_eval_set.csv")
support_eval = pd.read_csv("data/support_eval_set.csv")
susp_eval = pd.read_csv("data/suspensions_eval_set.csv")
throt_eval = pd.read_csv("data/throttling_eval_set.csv")

###Categorical Encoder
#runs an aggregation function on categorical features and one-hots them
#basically we might not want to throw away categorical stuff in full datset
def count_categorical(data, cat_column, cat_prefix, agg_func = "count", min_obs=50):
    data = data.copy()# so we don't mess with the origional dataframe
    #we wont be able to learn from categories with too few instances
    class_counts = data[cat_column].value_counts()
    allowed_classes = class_counts.index[(class_counts > 10).values]

    #replace the infrequent classes with 'OTHER' class label
    data.loc[~data[cat_column].isin(allowed_classes), cat_column] = "OTHER"

    #convert to wide format fill in negative cases (Nans) with 0
    a = data.groupby(["customer_id", cat_column]).apply(agg_func).iloc[:,0].reset_index(name="num")

    a=a.pivot(index='customer_id', columns=cat_column, values='num').fillna(0)
    #rename columns to reflect their original dataset
    a=a.rename(lambda s: cat_prefix + s, axis=1)
    return a



###Performing feature engineering on "act"¶
# Combine "act" by number of activations
ebb1 = ebb1.join(count_categorical(act1, "activation_channel", "act_chan_"))
act1 = act1.groupby('customer_id')['activation_date'].count().reset_index(name='number_activations')
ebb1 = ebb1.join(act1.set_index('customer_id'))
ebb2 = ebb2.join(count_categorical(act2, "activation_channel", "act_chan_"))
act2 = act2.groupby('customer_id')['activation_date'].count().reset_index(name='number_activations')
ebb2 = ebb2.join(act2.set_index('customer_id'))
ebb_eval = ebb_eval.join(count_categorical(act_eval, "activation_channel", "act_chan_"))
act_eval = (
    act_eval.groupby('customer_id')['activation_date'].count().reset_index(name='number_activations')
)
ebb_eval = ebb_eval.join(act_eval.set_index('customer_id'))



###Performing feature engineering on "auto"
# Combine "auto" by number of auto refills
auto1 = (
    auto1.groupby('customer_id')['auto_refill_enroll_date'].count().
    reset_index(name='number_auto_refill')
)
ebb1 = ebb1.join(auto1.set_index('customer_id'))
auto2 = (
    auto2.groupby('customer_id')['auto_refill_enroll_date'].count().
    reset_index(name='number_auto_refill')
)
ebb2 = ebb2.join(auto2.set_index('customer_id'))
auto_eval = (
    auto_eval.groupby('customer_id')['auto_refill_enroll_date'].count().
    reset_index(name='number_auto_refill')
)
ebb_eval = ebb_eval.join(auto_eval.set_index('customer_id'))


###Performing feature engineering on "deact"
# Combine "deact" by number of deactivations
deact1 = deact1.groupby('customer_id')['deactivation_date'].count().reset_index(name='number_deactivations')
ebb1 = ebb1.join(deact1.set_index('customer_id'))
deact2 = deact2.groupby('customer_id')['deactivation_date'].count().reset_index(name='number_deactivations')
ebb2 = ebb2.join(deact2.set_index('customer_id'))
deact_eval = deact_eval.groupby('customer_id')['deactivation_date'].count().reset_index(name='number_deactivations')
ebb_eval = ebb_eval.join(deact2.set_index('customer_id'))

### Performing feature engineering on "ivr"
# Combine "ivr" by number of calls
ivr1 = ivr1.groupby('customer_id')['call_start_time'].count().reset_index(name='number_calls')
ebb1 = ebb1.join(ivr1.set_index('customer_id'))
ivr2 = ivr2.groupby('customer_id')['call_start_time'].count().reset_index(name='number_calls')
ebb2 = ebb2.join(ivr2.set_index('customer_id'))
ivr_eval = ivr_eval.groupby('customer_id')['call_start_time'].count().reset_index(name='number_calls')
ebb_eval = ebb_eval.join(ivr2.set_index('customer_id'))


### Performing feature engineering on "loyalty"
# Combine "loyalty" by number of total quantity of points
loyalty1 = loyalty1.groupby('customer_id')['total_quantity'].sum().reset_index(name='total_quantity')
ebb1 = ebb1.join(loyalty1.set_index('customer_id'))
loyalty2 = loyalty2.groupby('customer_id')['total_quantity'].sum().reset_index(name='total_quantity')
ebb2 = ebb2.join(loyalty2.set_index('customer_id'))
loyalty_eval = loyalty_eval.groupby('customer_id')['total_quantity'].sum().reset_index(name='total_quantity')
ebb_eval = ebb_eval.join(loyalty_eval.set_index('customer_id'))


### Performing feature engineering on "network"
# Combine "network" by sum of voice minutes, total sms, total kb and hotspot kb
network1 = network1.groupby('customer_id')['voice_minutes','total_sms','total_kb','hotspot_kb'].sum()
ebb1 = ebb1.join(network1)
network2 = network2.groupby('customer_id')['voice_minutes','total_sms','total_kb','hotspot_kb'].sum()
ebb2 = ebb2.join(network2)
network_eval = network_eval.groupby('customer_id')['voice_minutes','total_sms','total_kb','hotspot_kb'].sum()
ebb_eval = ebb_eval.join(network_eval)

### Performing feature engineering on "react"
# Combine "react" by number of reactivations
react1 = react1.groupby('customer_id')['reactivation_date'].count().reset_index(name='number_reactivations')
ebb1 = ebb1.join(react1.set_index('customer_id'))
react2 = react2.groupby('customer_id')['reactivation_date'].count().reset_index(name='number_reactivations')
ebb2 = ebb2.join(react2.set_index('customer_id'))
react_eval = react_eval.groupby('customer_id')['reactivation_date'].count().reset_index(name='number_reactivations')
ebb_eval = ebb_eval.join(react_eval.set_index('customer_id'))


### Performing feature engineering on "redempt"
# Combine "redempt" by sum of revenue buckets
redempt1 = redempt1.groupby('customer_id')['revenues'].sum().reset_index(name='sum_revenues')
ebb1 = ebb1.join(redempt1.set_index('customer_id'))
redempt2 = redempt2.groupby('customer_id')['revenues'].sum().reset_index(name='sum_revenues')
ebb2 = ebb2.join(redempt2.set_index('customer_id'))
redempt_eval = redempt_eval.groupby('customer_id')['revenues'].sum().reset_index(name='sum_revenues')
ebb_eval = ebb_eval.join(redempt_eval.set_index('customer_id'))


### Performing feature engineering on "susp"
# # Combine "susp" by number of suspensions
susp1 = susp1.groupby('customer_id')['start_date'].count().reset_index(name='number_suspensions')
ebb1 = ebb1.join(susp1.set_index('customer_id'))
susp2 = susp2.groupby('customer_id')['start_date'].count().reset_index(name='number_suspensions')
ebb2 = ebb2.join(susp2.set_index('customer_id'))
susp_eval = susp_eval.groupby('customer_id')['start_date'].count().reset_index(name='number_suspensions')
ebb_eval = ebb_eval.join(susp2.set_index('customer_id'))

### Performing feature engineering on "throt"
# Combine "throt" by number of throttles
throt1 = throt1.groupby('customer_id')['throttled_date'].count().reset_index(name='number_throttles')
ebb1 = ebb1.join(throt1.set_index('customer_id'))
throt2 = throt2.groupby('customer_id')['throttled_date'].count().reset_index(name='number_throttles')
ebb2 = ebb2.join(throt2.set_index('customer_id'))
throt_eval = throt_eval.groupby('customer_id')['throttled_date'].count().reset_index(name='number_throttles')
ebb_eval = ebb_eval.join(throt_eval.set_index('customer_id'))



### Moving EBB to end
ebb = ebb1.pop('ebb_eligible')
ebb1['ebb_eligible'] = ebb
ebb = ebb2.pop('ebb_eligible')
ebb2['ebb_eligible'] = ebb

# Slight adjustment to "year" column for phone to filter out years above 2022
ebb1["year"] = pd.to_numeric(ebb1["year"])
under_years = ebb1["year"] <= 2022
avg_year = round(ebb1[under_years]["year"].sum()/under_years.sum())
ebb1.loc[(ebb1["year"] > 2022), "year"] = avg_year

ebb2["year"] = pd.to_numeric(ebb2["year"])
under_years = ebb2["year"] <= 2022
avg_year = round(ebb2[under_years]["year"].sum()/under_years.sum())
ebb2.loc[(ebb2["year"] > 2022), "year"] = avg_year



### Export to csv
from pathlib import Path  
filepath1 = Path('data/full_ebb_set1.csv')  
filepath1.parent.mkdir(parents=True, exist_ok=True)  
ebb1.to_csv(filepath1)  
filepath2 = Path('data/full_ebb_set2.csv')  
filepath2.parent.mkdir(parents=True, exist_ok=True) 
ebb2.to_csv(filepath2)
filepath_eval = Path('data/full_ebb_set_eval.csv')  
filepath_eval.parent.mkdir(parents=True, exist_ok=True) 
ebb_eval.to_csv(filepath_eval)
