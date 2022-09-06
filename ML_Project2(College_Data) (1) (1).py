#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')

clg_db = pd.read_csv('CollegeData.csv')
clg_db.tail()
#clg_db.info()


# In[77]:


#Creating a 

sns.set_style('darkgrid')
sns.lmplot('Room.Board','Grad.Rate',data=clg_db, hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)


# In[78]:


sns.set_style('darkgrid')
sns.lmplot('Outstate','F.Undergrad',data=clg_db, hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)


# In[79]:


sns.set_style('darkgrid')
g = sns.FacetGrid(clg_db,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# In[80]:


sns.set_style('darkgrid')
g = sns.FacetGrid(clg_db,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# In[81]:


clg_db[clg_db['Grad.Rate'] > 100]


# In[82]:


clg_db['Grad.Rate']['Cazenovia College'] = 100


# In[83]:


sns.set_style('darkgrid')
g = sns.FacetGrid(clg_db,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# In[84]:


X = clg_db[["Outstate","F.Undergrad"]]
np.random.seed(200)
Centroids = (X.sample(n=2))
plt.scatter(X["Outstate"],X["F.Undergrad"],c='black')
plt.scatter(Centroids["Outstate"],Centroids["F.Undergrad"],c='red')
plt.xlabel('Outstate')
plt.ylabel('F.Undergrad')
plt.show()


# In[85]:


diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Outstate"]-row_d["Outstate"])**2
            d2=(row_c["F.Undergrad"]-row_d["F.Undergrad"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(2):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["F.Undergrad","Outstate"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['F.Undergrad'] - Centroids['F.Undergrad']).sum() + (Centroids_new['Outstate'] - Centroids['Outstate']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["F.Undergrad","Outstate"]]


# In[86]:


color=['blue','cyan']
for k in range(2):
    clg_db=X[X["Cluster"]==k+1]
    plt.scatter(clg_db["Outstate"],clg_db["F.Undergrad"],c=color[k])
plt.scatter(Centroids["Outstate"],Centroids["F.Undergrad"],c='red')
plt.xlabel('Outstated')
plt.ylabel('F.Undergrad')
plt.show()


# In[64]:


clg_db = pd.read_csv('CollegeData.csv')
clg_db.head()
x=clg_db.iloc[:,:].values 
x
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
x[:,0]=labelencoder.fit_transform(x[:,0]) #To encode the state column and store it in the same position
x[:,0] #they will assign the number alphabetically
onehotencoder=OneHotEncoder() #convert each and everything into 0's and 1's
x=onehotencoder.fit_transform(x).toarray() #totally depends on probability
# #maximum value is provided with 1 and lowest in the form of 0
x
clg_db.head()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
clg_db_scaled = scaler.fit_transform(x)
pd.DataFrame(clg_db_scaled).describe()


# In[65]:


kmeans = KMeans(n_clusters=2, init='k-means++')


# In[20]:


kmeans.fit(clg_db_scaled)


# In[21]:


kmeans.inertia_


# In[22]:


kmeans = KMeans(n_jobs = -1, n_clusters = 2, init='k-means++')
kmeans.fit(clg_db_scaled)
pred = kmeans.predict(clg_db_scaled)


# In[23]:


frame = pd.DataFrame(clg_db_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[68]:


Y = clg_db[["Room.Board","Grad.Rate"]]
clg_db.head()


# In[70]:


plt.scatter(Y["Room.Board"],Y["Grad.Rate"],c='black')
plt.xlabel('Room.Board')
plt.ylabel('Grad.Rate')
plt.show()


# In[71]:


K=2


# In[72]:


np.random.seed(200)
Centroids = (Y.sample(n=K))
plt.scatter(Y["Room.Board"],Y["Grad.Rate"],c='black')
plt.scatter(Centroids["Room.Board"],Centroids["Grad.Rate"],c='red')
plt.xlabel('Room.Board')
plt.ylabel('Grad.Rate')
plt.show()


# In[73]:


diff = 1
j=0

while(diff!=0):
    YD=Y
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in YD.iterrows():
            d1=(row_c["Room.Board"]-row_d["Room.Board"])**2
            d2=(row_c["Grad.Rate"]-row_d["Grad.Rate"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        Y[i]=ED
        i=i+1

    C=[]
    for index,row in Y.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    Y["Cluster"]=C
    Centroids_new = Y.groupby(["Cluster"]).mean()[["Grad.Rate","Room.Board"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Grad.Rate'] - Centroids['Grad.Rate']).sum() + (Centroids_new['Room.Board'] - Centroids['Room.Board']).sum()
        print(diff.sum())
    Centroids = Y.groupby(["Cluster"]).mean()[["Grad.Rate","Room.Board"]]


# In[74]:


color=['blue','cyan']
for k in range(K):
    clg_db=Y[Y["Cluster"]==k+1]
    plt.scatter(clg_db["Room.Board"],clg_db["Grad.Rate"],c=color[k])
plt.scatter(Centroids["Room.Board"],Centroids["Grad.Rate"],c='red')
plt.xlabel('Room.Board')
plt.ylabel('Grad.Rate')
plt.show()


# In[ ]:




