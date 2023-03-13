#!/usr/bin/env python
# coding: utf-8

# # Expected Salary for fresher based on company employers salary

# ![LRM.jpg](attachment:LRM.jpg)

# # The scenario is you are a HR officer, you got a candidate with 5 years of experience.
# Then what is the best salary you should offer to him?”

# # # Importing Packages/Libraries
# 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# # Getting Details of Data set

# In[4]:


# Importing the dataset
google_kcdhanunjayInfosys = pd.read_csv("C:/Users/KCDHANUNJAY INFOSYS/AppData/Local/Programs/Python/Python312/salary_data.csv")


# In[5]:


# displaying salary_data
print(google_kcdhanunjayInfosys)


# In[6]:


#getting Datatypes of Columns
google_kcdhanunjayInfosys.dtypes


# In[7]:


#getting size of Dataframes(rows x Columns)
print(google_kcdhanunjayInfosys.size)


# In[8]:


#getting the number of rows and columns in the dataframe
google_kcdhanunjayInfosys.shape


# In[9]:


#getting the dimentions of the dataframe
google_kcdhanunjayInfosys.ndim


# In[8]:


#getting Summery of Dataset
google_kcdhanunjayInfosys.info()


# In[9]:


google_kcdhanunjayInfosys.info


# In[10]:


# head() Return the first 5 rows of the DataFrame by default.
google_kcdhanunjayInfosys.head()


# In[11]:


# head() Return the first n rows of the DataFrame.
google_kcdhanunjayInfosys.head(20)


# In[12]:


# head() Returning entire size of the DataFrame.
google_kcdhanunjayInfosys.head(np.size(google_kcdhanunjayInfosys))


# In[13]:


# tail() Return the last 5 rows of the DataFrame by default.
google_kcdhanunjayInfosys.tail()


# In[14]:


# tail() Return the last 10 rows of the DataFrame.
google_kcdhanunjayInfosys.tail(10)


# In[15]:


# tail() Returning entire rows of the DataFrame.
google_kcdhanunjayInfosys.tail(np.size(google_kcdhanunjayInfosys))


# In[10]:


print(google_kcdhanunjayInfosys.isna())


# In[18]:


print(google_kcdhanunjayInfosys.isna().astype(int))


# In[13]:


google_kcdhanunjayInfosys['YearsExperience']


# In[21]:


google_kcdhanunjayInfosys['YearsExperience'].astype(int)


# In[16]:


google_kcdhanunjayInfosys['YearsExperience'].isna().astype(int)


# In[12]:


google_kcdhanunjayInfosys['Salary']


# # # Linear Regression Initialization
# 

# In[29]:


#get a copy of dataset exclude last column
# X is first column of dataset
X = google_kcdhanunjayInfosys.iloc[:, :-1].values 


# In[30]:


X


# In[31]:


#get array of dataset in column 1st
y = google_kcdhanunjayInfosys.iloc[:, 1].values 


# In[32]:


y


# # # Spilting data

# In[35]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# # # # Fitting Simple Linear Regression to the Training set
# 
# # Predicting the result of 5 Years Experience
# 

# In[36]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[37]:


# Predicting the result of 5 Years Experience
from sklearn.linear_model import LinearRegression
y_pred = regressor.predict(X_test)
y_pred


# In[38]:


# Predicting the result of 5 Years Experience
y_pred = regressor.predict([[5]])
y_pred


# # #  intercept and slope of a simple linear regression model in Python using scikit-learn

# In[21]:


# print the intercept and slope
from sklearn.linear_model import LinearRegression
print("Intercept:",regressor.intercept_)
print("Slope:",regressor.coef_)


# In[39]:


print(regressor.get_params())


# # # Visualizing the Training set results
# 

# In[40]:


# Visualizing the Training set results
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()


# # Visualizing the Test set results

# In[42]:


# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()


# In[ ]:




