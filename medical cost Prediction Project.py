#!/usr/bin/env python
# coding: utf-8

# In[2]:


#A health insurance company can only make money if it collects more than it spends on the medical care of its beneficiaries. On the other hand, even though some conditions are more prevalent for certain segments of the population, medical costs are difficult to predict since most money comes from rare conditions of the patients. The objective of this article is to accurately predict insurance costs based on people’s data, including age, Body Mass Index, smoking or not, etc. Additionally, we will also determine what the most important variable influencing insurance costs is. These estimates could be used to create actuarial tables that set the price of yearly premiums higher or lower according to the expected treatment costs. This is a regression problem.


# In[ ]:


#Importig all modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[3]:


#importing dataset
data=pd.read_csv("D:\medical cost prediction\medical cost.csv")
data


# In[4]:


#first five values in the dataset
data.head()


# In[5]:


#last 5 values in the dataset
data.tail()


# In[6]:


checking the empty cells
data.isnull().sum()


# In[7]:


#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
data.sex


# In[8]:


# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
data.smoker


# In[9]:


#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)
data.region


# In[10]:


#correlation
data.corr()['charges'].sort_values()


# In[11]:


sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", data=data)


# In[12]:


pl.figure(figsize=(10,6))
ax = sns.scatterplot(x='bmi',y='charges',data=data,palette='magma',hue='smoker')
ax.set_title('Scatter plot of charges and bmi')

sns.lmplot(x="bmi", y="charges", hue="smoker", data=data, palette = 'magma',height= 8)


# In[13]:


sns.catplot(x="children", kind="count", palette="ch:.25", data=data,height= 6)


# In[14]:


#Independent variables used to predict cost
x= data.drop(['charges','region'], axis = 1)
x


# In[15]:


#dependent variable(cost prediction)
y= data.charges
y


# In[16]:


#Accuracy of the project
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.21,random_state=0)
lr = LinearRegression().fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print(lr.score(x_test,y_test))


# In[17]:


#predicting the cost
a= lr.predict([[21,0,25.800,0,0]])
a[0]


# In[20]:


#predicting the cost
b=lr.predict([[32,1,30.97,3,0]])
b[0]

