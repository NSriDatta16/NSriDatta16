#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


df_train=pd.read_csv('train.csv',index_col='PassengerId')
print(df_train.shape)
df_train.head()


# In[4]:


df_test=pd.read_csv('test.csv',index_col='PassengerId')
print(df_test.shape)
df_test.head()


# In[5]:


X_Test = df_test.copy()
X_Test.head()


# In[6]:


y_test=pd.read_csv('sample_submission.csv',index_col='PassengerId')
y_test.head()


# In[7]:


features_to_drop = ['Name']
df_train.drop(features_to_drop, axis=1, inplace=True)
X_Test.drop(features_to_drop, axis=1, inplace=True)


# In[8]:


X_Train = df_train.copy()
y_train = X_Train.pop('Transported')
y_train.replace([False, True], [0, 1], inplace=True)


# In[9]:


X_Train.head()


# In[10]:


y_train.head().to_frame()


# In[11]:


X_Train.dtypes.to_frame()


# In[12]:


X_Train['CryoSleep'].replace([False, True], [0, 1], inplace=True)
X_Train['CryoSleep'].fillna(0, inplace=True)
X_Train['CryoSleep'] = X_Train['CryoSleep'].astype(int)

X_Test['CryoSleep'].replace([False, True], [0, 1], inplace=True)
X_Test['CryoSleep'].fillna(0, inplace=True)
X_Test['CryoSleep'] = X_Test['CryoSleep'].astype(int)


# In[13]:


X_Train.drop('Cabin',axis=1,inplace=True)
X_Test.drop('Cabin',axis=1,inplace=True)


# In[14]:


X_Train.head()


# In[15]:


X_Train['HomePlanet'].value_counts()


# In[16]:


X_Train['Destination'].value_counts()


# In[17]:


X_Test.head()


# In[18]:


X_Train.isnull().sum()


# In[19]:


X_Train.dtypes


# In[20]:


X_Test.dtypes


# In[21]:


# Filling Missing Values with zeros on train dataset
X_Train['RoomService'].fillna(0, inplace=True)
X_Train['FoodCourt'].fillna(0, inplace=True)
X_Train['ShoppingMall'].fillna(0, inplace=True)
X_Train['Spa'].fillna(0, inplace=True)
X_Train['VRDeck'].fillna(0, inplace=True)

# Filling Missing Values with zeros on validation dataset
X_Test['RoomService'].fillna(0, inplace=True)
X_Test['FoodCourt'].fillna(0, inplace=True)
X_Test['ShoppingMall'].fillna(0, inplace=True)
X_Test['Spa'].fillna(0, inplace=True)
X_Test['VRDeck'].fillna(0, inplace=True)


# In[22]:


X_Train['VIP'].replace([False, True], [0, 1], inplace=True)
X_Train['VIP'].fillna(0, inplace=True)
X_Train['VIP'] = X_Train['VIP'].astype(int)

X_Test['VIP'].replace([False, True], [0, 1], inplace=True)
X_Test['VIP'].fillna(0, inplace=True)
X_Test['VIP'] = X_Test['VIP'].astype(int)


# In[23]:


X_Train['Age'].fillna(X_Train['Age'].mean(), inplace=True)
X_Train['Age'] = X_Train['Age'].astype(int)

#X_test['HomePlanet'].fillna(X_test['HomePlanet'].value_counts().idxmax(), inplace=True)
#X_test['Destination'].fillna(X_test['Destination'].value_counts().idxmax(), inplace=True)
X_Test['Age'].fillna(X_Test['Age'].mean(), inplace=True)
X_Test['Age'] = X_Test['Age'].astype(int)


# In[24]:


X_Train.isnull().sum()


# In[25]:


X_Test.isnull().sum()


# In[26]:


X_Train.dtypes


# In[27]:


X_Test.dtypes


# In[28]:


X_Train.head()


# In[29]:


X_Test.head()


# In[30]:


X_Train['HomePlanet'].value_counts()


# In[31]:


X_train = pd.get_dummies(X_Train, columns=['HomePlanet','Destination'])
X_train.head()


# In[32]:


X_test = pd.get_dummies(X_Test, columns=['HomePlanet','Destination'])
X_test.head()


# In[33]:


X_train.dtypes


# In[34]:


X_test.dtypes


# In[35]:


X_train['HomePlanet_Earth'] = X_train['HomePlanet_Earth'].astype(int)
X_train['HomePlanet_Europa'] = X_train['HomePlanet_Europa'].astype(int)
X_train['HomePlanet_Mars'] = X_train['HomePlanet_Mars'].astype(int)
X_train['Destination_55 Cancri e'] = X_train['Destination_55 Cancri e'].astype(int)
X_train['Destination_PSO J318.5-22'] = X_train['Destination_PSO J318.5-22'].astype(int)
X_train['Destination_TRAPPIST-1e'] = X_train['Destination_TRAPPIST-1e'].astype(int)

X_test['HomePlanet_Earth'] = X_test['HomePlanet_Earth'].astype(int)
X_test['HomePlanet_Europa'] = X_test['HomePlanet_Europa'].astype(int)
X_test['HomePlanet_Mars'] = X_test['HomePlanet_Mars'].astype(int)
X_test['Destination_55 Cancri e'] = X_test['Destination_55 Cancri e'].astype(int)
X_test['Destination_PSO J318.5-22'] = X_test['Destination_PSO J318.5-22'].astype(int)
X_test['Destination_TRAPPIST-1e'] = X_test['Destination_TRAPPIST-1e'].astype(int)


# In[36]:


X_train['Transported']=y_train
X_test['Transported']=y_test


# In[37]:


df_combined = pd.concat([X_train, X_test])
df_combined.head()


# In[38]:


X=df_combined.copy()
y=X.pop('Transported')
X.head()


# In[39]:


X.columns


# In[40]:


print(X.shape,y.shape)


# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                            n_redundant=0, n_clusters_per_class=1, random_state=42)

# Create a Random Forest Classifier with 100 trees
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the data
rfc.fit(X,y)

# Get the feature importances
feature_importances = rfc.feature_importances_

# Create a dataframe with feature importances and column names
df_importances = pd.DataFrame({'Feature': list(range(1, X.shape[1]+1)), 
                               'Column Name': ['Col1', 'Col2', 'Col3', 'Col4', 'Col5'], 
                               'Importance': feature_importances})

# Sort the dataframe by feature importance
df_importances = df_importances.sort_values('Importance', ascending=False)

# Print the dataframe
print(df_importances)


# In[42]:


from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X,y,train_size=0.75, test_size=0.25)

tpot= TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
tpot.fit(X_train_full, y_train_full)
print(tpot.score(X_test_full, y_test_full))


# In[102]:


import pandas as pd
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#model= BernoulliNB(alpha=0.01,fit_prior=False)#LinearSVC(C=0.5,dual=True, loss='hinge', penalty='l2', tol=1e-05), alpha=0.01,fit_prior=False)

model = SGDClassifier(alpha=0.001, eta0=0.1, fit_intercept=False, l1_ratio=1.0, learning_rate='invscaling', loss='perceptron', penalty='elasticnet', power_t=0.5)

#model = SGDClassifier(loss='log', alpha=0.001, learning_rate='constant', eta0=0.01, max_iter=1000)

#model = ExtraTreesClassifier()

#model = GradientBoostingClassifier()


# fit the model to the training data
model.fit(X_train_full, y_train_full)

# make predictions on the test data
y_pred = model.predict(X_test_full)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test_full, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#print(accuracy)


# In[ ]:




