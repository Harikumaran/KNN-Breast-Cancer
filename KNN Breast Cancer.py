#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split


# In[10]:


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)


# In[11]:


X = np.array(df.drop(['class'],1))
y = np.array(df['class'])


# In[12]:


X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[13]:


clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)


# In[29]:


accuracy = clf.score(X_test,y_test)
print(accuracy)


# In[20]:


example_measures = np.array([1010101,4,2,1,1,1,2,3,2,1,1])
example_measures = example_measures.reshape(1,-1)
prediction = clf.predict(example_measures)
print(prediction)


# In[ ]:




