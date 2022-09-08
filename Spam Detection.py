#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
data.head()


# In[3]:


#From the datset we only need class and message to train the model
data=data[["class","message"]]


# In[4]:


#Splitting the data into training and testing set, 

x=np.array(data["message"])
y=np.array(data["class"])
cv = CountVectorizer() #transforms text into vector based on freaquency of each word
X=cv.fit_transform(x)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

clf = MultinomialNB() #classification with dicreate features
clf.fit(X_train,y_train)


# In[11]:


sample = input('Enter a message:')
data = cv.transform([sample]).toarray()
print(clf.predict(data))


# In[ ]:




