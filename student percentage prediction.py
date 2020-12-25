#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df =pd.read_csv('http://bit.ly/w-data')
df.head(10)


# In[5]:


df.plot('Hours','Scores',style='o')
plt.title("Hours vs Percentage")
plt.xlabel("Hours studied")
plt.ylabel("percentage scored")
plt.show()


# In[6]:


X=df[["Hours"]].values
y=df["Scores"].values


# In[7]:


X


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)


# In[9]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)


# In[10]:


line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.xlabel("Study Hours")
plt.ylabel("Percentage Scored")
plt.title("Best fit line")
plt.show()


# In[11]:


X_test


# In[12]:


y_pred=regressor.predict(X_test)


# In[13]:


y_pred


# In[14]:


df=pd.DataFrame({"Hours Studied":X_test.reshape(-1), "Actual Percentage":y_test, "Predicted Percentage":y_pred})


# In[15]:


df


# In[16]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours ={}".format(hours[0][0]))
print("Predicted Score = {}".format(own_pred[0]))


# In[17]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))


# In[ ]:




