#!/usr/bin/env python
# coding: utf-8
 *************HEART DISEASE PREDICTION MODEL**************
# In[38]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


heart_db = pd.read_csv("heart_disease.csv")
heart_db.head()
#heart_db.info()

#Removing NaN values:
heart_db.dropna(axis=0, inplace=True)
print(heart_db.head(), heart_db.shape)
print(heart_db.AHD.value_counts())


# In[42]:


#Counting the number of patients affected with AHD:
heart_db["AHD"]=heart_db.AHD.replace({"No":0,"Yes":1})
plt.figure(figsize=(7,5))
sns.countplot(x='AHD', data = heart_db, palette="BuGn_r")


# In[48]:


#Splitting of data into training and tesing sets

X=np.asarray(heart_db[['Age','Sex','ResrBp','Chol','Fbs','RestECG','MaxHR','EXAng','Slope','Ca']])
y=np.asarray(heart_db['AHD'])
x
y
                       


# In[49]:


#Normalizing the data :
X = preprocessing.StandardScaler().fit(X).transform(X)


# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state =4)
print(f"Training Set : {X_train.shape, y_train.shape}")
print(f"Testing Set : {X_test.shape, y_test.shape}")


# In[58]:


#Modelling the dataset :
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)


# In[67]:


#Evalution and accuracy :
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm, columns = ['Predicted : 0', 'Predicted : 1'], index = ['Actual : 0','Actual : 1'])

plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "YlGnBu")
plt.show()


# In[79]:


print(f"The details for confusion matrix is:\n{classification_report(y_test, y_pred)}")


# In[76]:


def accuracy(y_test, y_pred):
    accuracy=np.sum(y_test==y_pred)/len(y_test)
    return accuracy


# In[78]:


print("Accuracy is : ", accuracy(y_test, y_pred))


# In[ ]:




