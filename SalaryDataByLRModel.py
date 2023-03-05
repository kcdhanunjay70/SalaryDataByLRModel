#!/usr/bin/env python
# coding: utf-8

# # The scenario is you are a HR officer, you got a candidate with 5 years of experience.
# Then what is the best salary you should offer to him?”

# # # Importing Packages/Libraries
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# # Getting Details of Data set

# In[2]:


# Importing the dataset
google_kcdhanunjayInfosys = pd.read_csv("C:/Users/KCDHANUNJAY INFOSYS/AppData/Local/Programs/Python/Python312/salary_data.csv")


# In[3]:


# displaying salary_data
print(google_kcdhanunjayInfosys)


# In[4]:


#getting Datatypes of Columns
google_kcdhanunjayInfosys.dtypes


# In[5]:


#getting size of Dataframes(rows x Columns)
print(google_kcdhanunjayInfosys.size)


# In[6]:


#getting the number of rows and columns in the dataframe
google_kcdhanunjayInfosys.shape


# In[7]:


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


# # # Linear Regression Initialization
# 

# In[16]:


#get a copy of dataset exclude last column
# X is first column of dataset
X = google_kcdhanunjayInfosys.iloc[:, :-1].values 


# In[17]:


X


# In[18]:


#get array of dataset in column 1st
y = google_kcdhanunjayInfosys.iloc[:, 1].values 


# In[19]:


y


# # # Spilting data

# In[20]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# # # # Fitting Simple Linear Regression to the Training set
# 
# # Predicting the result of 5 Years Experience
# 

# In[21]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[22]:


# Predicting the result of 5 Years Experience
from sklearn.linear_model import LinearRegression
hgt = google_kcdhanunjayInfosys.YearsExperience.values.reshape(-1,1)
wgt = google_kcdhanunjayInfosys.Salary.values.reshape(-1,1)
y_pred = regressor.predict(X_test)
y_pred


# In[23]:


# Predicting the result of 5 Years Experience
hgt = google_kcdhanunjayInfosys.YearsExperience.values.reshape(-1,1)
wgt = google_kcdhanunjayInfosys.Salary.values.reshape(-1,1)
y_pred = regressor.predict([[5]])
y_pred


# # #  intercept and slope of a simple linear regression model in Python using scikit-learn

# In[24]:


# print the intercept and slope
from sklearn.linear_model import LinearRegression
print("Intercept:",regressor.intercept_)
print("Slope:",regressor.coef_)


# In[25]:


print(regressor.get_params())


# # # Visualizing the Training set results
# 

# In[26]:


# Visualizing the Training set results
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()

# In[27]:

# # Visualizing the Test set results

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()
