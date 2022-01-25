#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from six import StringIO  
from IPython.display import Image 
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 


# In[26]:


df = pd.read_csv("Diabetes.csv")


# In[27]:


df


# In[28]:


df.info()


# In[29]:


df.columns


# In[30]:


df.drop(columns=['PhysHlth', 'DiffWalk', 'Unnamed: 0', "Education"], inplace= True)


# In[31]:


plt.figure(figsize=(16, 16))
sns.heatmap(df.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1);


# In[32]:


plt.figure(figsize=(16, 16))
sns.pairplot(df.sample(1000));


# In[33]:


df.columns


# In[ ]:





# In[34]:


#Divide dataset into features and Traget variables
X= df.loc[:, "HighBP": "Income"]
y = df["Diabetes_012"]


# In[35]:


def accuracy(actuals, preds):
    return np.mean(actuals == preds)

def precision(actuals, preds):
    tp = np.sum((actuals == 1) & (preds == 1))
    fp = np.sum((actuals == 0) & (preds == 1))
    return tp / (tp + fp)

def recall(actuals, preds):
    tp = np.sum((actuals == 1) & (preds == 1))
    fn = np.sum((actuals == 1) & (preds == 0))
    return tp / (tp + fn)

def F1(actuals, preds):
    p, r = precision(actuals, preds), recall(actuals, preds)
    return 2*p*r / (p + r)


# In[36]:


from sklearn.preprocessing import StandardScaler
target = df.Diabetes_012.value_counts(normalize=True) 
print(target)
sns.barplot(target.index, target.values)
plt.title('Diabetes Data Ratio')
plt.ylabel('Percentage of Data', fontsize=12);


# # Split Data into Train and Test

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
X_train.shape, y_train.shape


# # Over Sample Data to Fix Imbalance

# In[39]:


import imblearn.over_sampling

# Set up for the ration argument of RandomOverSampler Initialization 
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)

ratio = {1: n_pos *4, 0: n_neg}


# In[40]:


ROS = imblearn.over_sampling.RandomOverSampler(sampling_strategy = ratio, random_state = 42)

X_train_rs, y_train_rs = ROS.fit_resample(X_train, y_train)


# _____________________________________________________________________________________________________________________

# # RANDOM Forest Algorithm

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#create a Gaussian Classifier
random_forest_clf=rfc = RandomForestClassifier(n_estimators=100, n_jobs = -1,class_weight="balanced",random_state =50 , min_samples_leaf = 10)
rfc.fit(X_train_rs,y_train_rs)
y_pred_rfc = random_forest_clf.predict(X_test)


# In[42]:


print("confusion matrix: \n\n", 
      confusion_matrix(y_test, y_pred_rfc))

print(classification_report(y_test, y_pred_rfc))


# In[43]:


scores = cross_val_score(random_forest_clf, X_train_rs, y_train_rs, cv=10)
print('Cross-Validation Accuracy Scores', scores)


# In[44]:


scores = pd.Series(scores)
scores.min(), scores.mean(), scores.max()


# In[45]:


print('Random Forest validation metrics: \n Accuracy: %.4f \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (accuracy(y_test, random_forest_clf.predict(X_test)), 
         precision(y_test, random_forest_clf.predict(X_test)), 
         recall(y_test, random_forest_clf.predict(X_test)),
         F1(y_test, random_forest_clf.predict(X_test))
        )
     )


# In[46]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# # Random Hyperparameter Grid

# In[47]:


from sklearn.model_selection import RandomizedSearchCV

#Number of Trees in random forest
n_estimators= [int(x) for x in np.linspace(start =200, stop = 2000, num = 10)]

#Number of features to consider at every split 
max_features = ["auto", "sqrt"]

#Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10,110, num=11)]
max_depth.append(None)

#minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

#minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

#Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)
#On each iteration, the algorithm will choose a difference combination of the features.


# # Random Seach Training 
# 

# In[ ]:


# Use the random grod to search for best hyperparameters
#First create the base model to tune
rf = RandomForestRegressor()

#Random search of parameters, using 3 fold cross validation 
#Search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                              n_iter = 10,
                              cv=3,
                              verbose =2,
                              random_state =42,
                              n_jobs = -1)

#Fit the random search model
rf_random.fit(X_train_rs, y_train_rs)


# # View Best Paramters from fitting the Random Search 

# In[ ]:


rf_random.best_params_


# In[ ]:


random_forest_clf=rfc = RandomForestClassifier(n_estimators=2000, 
                                               min_samples_split= 5,
                                               min_samples_leaf=  2,
                                               max_features= "auto",
                                               max_depth = 50,
                                               bootstrap= True)
rfc.fit(X_train_rs, y_train_rs)
y_pred_rfc = random_forest_clf.predict(X_test)


# In[ ]:





# In[ ]:


print("confusion matrix: \n\n", 
      confusion_matrix(y_test, y_pred_rfc))

print(classification_report(y_test, y_pred_rfc))


# In[ ]:


scores = cross_val_score(random_forest_clf, X_train_rs, y_train_rs, cv=10)
print('Cross-Validation Accuracy Scores', scores)


# In[ ]:


scores = pd.Series(scores)
scores.min(), scores.mean(), scores.max()


# In[ ]:


print('Random Forest validation metrics: \n Accuracy: %.4f \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (accuracy(y_test, random_forest_clf.predict(X_test)), 
         precision(y_test, random_forest_clf.predict(X_test)), 
         recall(y_test, random_forest_clf.predict(X_test)),
         F1(y_test, random_forest_clf.predict(X_test))
        )
     )


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
random_forest_clf.feature_importances_


# In[ ]:


df.columns


# In[ ]:


feature_names= df.columns.drop(['Unnamed: 0', 'Diabetes_012'])


# In[ ]:


import time

start_time = time.time()
importances = random_forest_clf.feature_importances_
std = np.std([
    tree.feature_importances_ for tree in random_forest_clf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")


# In[ ]:


forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")


# In[ ]:


#pairplot, high lineraly correlation makes model unstable. Relationships engneer


# In[ ]:


from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(
    random_forest_clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)


# In[ ]:


fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()


# In[ ]:


make_confusion_matrix(random_forest_clf, threshold=0.5)


# _____________________________________________________________________________________________________________________
