#!/usr/bin/env python
# coding: utf-8

# # Model Training

# In[1]:


from typing import Optional, Iterable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import LocalOutlierFactor
from datetime import datetime

pd.options.mode.chained_assignment = None  # default='warn'


# ## Load Data

# In[2]:


ebb_1 = pd.read_csv("data/full_ebb_set1.csv").set_index("customer_id")
ebb_2 = pd.read_csv("data/full_ebb_set2.csv").set_index("customer_id")

model_df = pd.concat([ebb_1, ebb_2], axis=0)


# ## Feature/Target split

# Use this section to do split between training feature dataframe and target array. 
# * Please add all the categorical features to categorical features list below
# * Please add all the numeric features in the numeric features list below

# In[3]:


target_column = "ebb_eligible"

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
    "number_activations",
    "number_auto_refill",
    "number_deactivations",
    "number_calls",
    "year",
    "total_revenues_bucket",
    "total_sms",
    "total_kb",
    "hotspot_kb",
    "number_suspensions",
    "number_throttles",
]

X = model_df[
    [
        col for col in model_df.columns 
        if ((col in categorical_features) or (col in numerical_features))
    ]
]
y = model_df[target_column]


# ## Data Encoding

# ### Encoding pipeline class

# In[4]:


class EncodingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features: list, numerical_features: list):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
    def _impute_missing_values_categorical(self, X):
        # Set missing categorical values to unknown
        X.loc[:, self.categorical_features] = (
            X.astype(object).
            loc[:, self.categorical_features].
            where(
                pd.notnull(X[self.categorical_features]), "Unknown"
            )
        )
        return X
        
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        required_columns = set(self.categorical_features).union(set(self.numerical_features))
        missing_columns = required_columns.difference(set(X.columns))
        assert len(missing_columns) == 0, f"Missing required columns {missing_columns}"
        
        X = self._impute_missing_values_categorical(X)
        
        # One hot encode categorical features
        self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoder.fit(X[self.categorical_features].astype(str))
        self.encoded_feature_names = self.encoder.get_feature_names_out(
            self.categorical_features
        )
        
        # Use simple imputer for numerical values
        self.numeric_imputer = SimpleImputer()
        self.numeric_imputer.fit(X[self.numerical_features])
        
        return self
        
    def transform(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        X = self._impute_missing_values_categorical(X)
        
        # One hot encode categorical features
        X.loc[:, self.encoded_feature_names] = (
            self.encoder.transform(X[self.categorical_features].astype(str))
        )
        # Delete original categorical values
        for col in self.categorical_features:
            if col in X:
                del X[col]
        
        # Use simple imputer for numerical values
        X.loc[:, self.numerical_features] = (
            self.numeric_imputer.transform(X[self.numerical_features])
        )
        
        # Filter required columns
        final_columns = list(set(self.encoded_feature_names).union(self.numerical_features))
        
        return X[final_columns]


# ### Encode values

# In[5]:


pipeline = EncodingPipeline(categorical_features, numerical_features)

X_encoded = pipeline.fit_transform(X.copy())

# Need to convert y = 1 to -1 for one class svm and missing values to 1
y[y == 1] = -1
y[pd.isna(y)] = 1


# ## Set up random search

# In[6]:


parameter_grid_isolation_forest = {
    "n_estimators": list(map(int, np.logspace(1.0, 3.0, num=10))),
}

parameter_knn = {
    "n_neighbors": list(map(int, np.logspace(1, 2, num=10))),
    "leaf_size": list(map(int, np.logspace(1.5, 2, 10))),    
}


# ## Fit a classifier

# In[7]:


isolation_forest_clf = IsolationForest()
knn_clf = LocalOutlierFactor(novelty=True)
models = []
n_iterations = 5

# Random forest 
forest_random_search = RandomizedSearchCV(
    isolation_forest_clf, 
    parameter_grid_isolation_forest, 
    n_iter=n_iterations, 
    cv=5, 
    n_jobs=1, 
    verbose=5,
    scoring="accuracy",
    return_train_score=True,
)
forest_random_search.fit(X_encoded.values[:5000], y[:5000])
models.append(forest_random_search)

# K nearest neighbours
knn_random_search = RandomizedSearchCV(
    knn_clf, 
    parameter_knn, 
    n_iter=n_iterations, 
    cv=5, 
    n_jobs=1, 
    verbose=5,
    scoring="accuracy",
    return_train_score=True,    
)
knn_random_search.fit(X_encoded.values[:5000], y[:5000])
models.append(knn_random_search)


# ## Train Predictions

# In[8]:


# Choose best model from list of trained randomsearch models
model_score = 0
best_score = 0
for model in models:
    model_score = f1_score(y, model.predict(X_encoded.values))
    print(f"Model score for {model} is: {model_score}")
    if model_score > best_score:
        final_model = model
        best_score = model_score

# Fit final model with entire dataset
print("Fitting classifier")
final_clf = final_model.best_estimator_
final_clf.fit(X_encoded.values, y)
        
print(f"Best model is {final_clf}.\nBest F1 score training: {best_score}")


# ## Eval Set

# In[9]:


eval_df = pd.read_csv("data/full_ebb_set_eval.csv").set_index("customer_id")
eval_df_encoded = pipeline.transform(eval_df)

# Predict on eval set
eval_predictions = final_clf.predict(eval_df_encoded.values)

# Set predictions to 1 and 0
eval_predictions[eval_predictions == 1] = 0
eval_predictions[eval_predictions == -1] = 1


# In[10]:


eval_predictions_df = pd.DataFrame({"customer_id": eval_df.index, "ebb_eligible": eval_predictions})


# In[11]:


current_date = datetime.now().strftime("%Y-%m-%d")
eval_predictions_df.to_csv(f"submission/{current_date}.csv", index=False)


# In[ ]:




