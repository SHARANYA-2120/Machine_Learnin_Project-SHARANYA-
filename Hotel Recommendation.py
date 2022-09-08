#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('wordnet')
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval

data = pd.read_csv("Hotel_Reviews.csv")
data.head()


# In[2]:


#Extracting different countries in the dataset and storing it in the new column
data["countries"]=data.Hotel_Address.apply(lambda x:x.split(' ')[-1])
print(data.countries.unique())


# In[3]:


#Dropping unnecessary columns:
data.drop(['Additional_Number_of_Scoring',
       'Review_Date','Reviewer_Nationality',
       'Negative_Review', 'Review_Total_Negative_Word_Counts',
       'Total_Number_of_Reviews', 'Positive_Review',
       'Review_Total_Positive_Word_Counts',
       'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score',
       'days_since_review', 'lat', 'lng'],1,inplace=True)


# In[4]:


#converting the strings of list into normal list and then applying it to the 'Tags' column in dataset

def impute(column):
    column = column[0]
    if(type(column)!=list):
        return "".join(literal_eval(column))
    else: 
        return column
    data["Tags"]=data[["Tags"]].apply(impute,axis=1)
    print(data.head())
    


# In[5]:


data['countries'] = data['countries'].str.lower()
data['Tags'] = data['Tags'].str.lower()


# In[6]:


import nltk
nltk.download('punkt')
nltk.download('stopwords') #(downloads a file with english stopwords) stopwords : articles, pronouns,prepositions etc..
nltk.download('omw-1.4')

def recommend_hotel(location, description):  #function to recommend a hotel as per customer requirements
    description = description.lower()
    word_tokenize(description)
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer() #Links words with similar meanings to one word.
    filtered  = {word for word in description if not word in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))
        
    #print(filtered_set)
    country = data[data['countries']==location.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    list1 = []; list2 = []; cos = [];
    
    for i in range(country.shape[0]):
        temp_token = word_tokenize(country["Tags"][i]) #Splits the sentence into tockens or words.
        temp_set = [word for word in temp_token if not word in stop_words]
#         print(temp_set)
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))
        #print(temp2_set)
        vector = temp2_set.intersection(filtered_set) #Returns similarity between two or more sets
        cos.append(len(vector))
    country['similarity']=cos
    country = country.sort_values(by='similarity', ascending=False)
    country.drop_duplicates(subset='Hotel_Name', keep='first', inplace=True)
    country.sort_values('Average_Score', ascending=False, inplace=True)
    country.reset_index(inplace=True)
    return country[["Hotel_Name", "Average_Score", "Hotel_Address"]].head()


# In[7]:


location=input('Please enter your location :')
description=input('Please mention the occassion for the travel :')
recommend_hotel(location,description)


# In[ ]:




