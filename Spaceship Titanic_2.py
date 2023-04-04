#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Load The Datasets
df=pd.read_csv('train.csv',index_col="PassengerId")

#Check the Size of the Dataset , no. of rows and columns.
print(df.shape)

#Check the top rows of the dataset using head() command. 
df.head()


# In[3]:


num=df.select_dtypes(include=np.number)
df_num=num.apply(lambda rec:rec-rec.mean()/rec.std(),axis=0)

cat=df.select_dtypes(include='object')
df_cat=pd.get_dummies(cat,drop_first=True)


# In[4]:


#Load the Test Dataset.
df1=pd.read_csv('test.csv')

#Print the size od the dataset
print(df1.shape)

df1.head()


# In[5]:


df2=pd.read_csv('sample_submission.csv')
print(df2.shape)
df2.head()


# In[6]:


#Check For the Null Values in each column.
df.isnull().sum()


# In[7]:


df['HomePlanet'].fillna(df['HomePlanet'].mode()[0], inplace=True)


# In[8]:


df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)


# In[9]:


df['Destination'].fillna(df['Destination'].mode()[0], inplace=True)


# In[10]:


df['Age'].fillna(df['Age'].mean(), inplace=True)


# In[11]:


df['RoomService'].fillna(df['RoomService'].mean(), inplace=True)


# In[12]:


df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=True)


# In[13]:


df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=True)


# In[14]:


df['Spa'].fillna(df['Spa'].mean(), inplace=True)


# In[15]:


df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=True)


# In[16]:


df.isnull().sum()


# In[17]:


#Check For the Data Types of each columns.
df.dtypes.to_frame()


# In[18]:


df.dtypes.to_frame()


# In[19]:


df.drop('Name',inplace=True,axis=1)


# In[20]:


df.columns


# In[21]:


df['HomePlanet'].value_counts()


# In[22]:


df['Cabin'].value_counts().head()


# In[23]:


df['Destination'].value_counts()


# In[24]:


df['VIP'].value_counts()
df['VIP'] = df['VIP'].fillna(0).astype(int)
#Change the Categorical Target Column to Numerical by changing its Datatype and Values.
df['VIP'] = df['VIP'].astype(int)

#Check the Column in DataSet Whether it is changed or not.
df['VIP'].astype(int)

df['VIP'].value_counts()


# In[25]:


df['RoomService'].value_counts()


# In[26]:


df['FoodCourt'].value_counts()


# In[27]:


df['ShoppingMall'].value_counts()


# In[28]:


df['Spa'].value_counts()


# In[29]:


df['VRDeck'].value_counts()


# In[30]:


df['Transported'].value_counts()

#Change the Categorical Target Column to Numerical by changing its Datatype and Values.
df['Transported'] = df['Transported'].astype(int)

#Check the Column in DataSet Whether it is changed or not.
df['Transported'].astype(int)

df['Transported']


# In[31]:


print(df['CryoSleep'].value_counts())
df['CryoSleep'] = df['CryoSleep'].fillna(0).astype(int)
df['CryoSleep'] = df['CryoSleep'].astype(int)
df['CryoSleep'].astype(int)
df['CryoSleep'].value_counts()


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[33]:


df.head()


# In[34]:


# Categorical Features Stats
df.describe(include=['object'])


# In[36]:


# Numerical Features Distribution
df.hist(bins=20, figsize=(20, 20));


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


X = df.drop(['Transported'], axis=1)  # Independent variables
y = df['Transported']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform data preprocessing (e.g., imputation, scaling, encoding)

# Train the logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:




