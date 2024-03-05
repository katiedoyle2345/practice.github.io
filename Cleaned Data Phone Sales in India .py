#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the cleaned data
data_path = '/Users/katie/Downloads/Cleaned_Sales.csv'
cleaned_data = pd.read_csv(data_path)

# Display the first few rows of the data to understand its structure
cleaned_data.head()


# In[2]:


# Remove '€' symbol and convert to float
cleaned_data['SellingPrice_EUR'] = cleaned_data['SellingPrice_EUR'].str.replace('€', '').astype(float)
cleaned_data['OriginalPrice_EUR'] = cleaned_data['OriginalPrice_EUR'].str.replace('€', '').astype(float)

# Convert 'DiscountPercentage' to float after removing '%'
cleaned_data['DiscountPercentage'] = cleaned_data['DiscountPercentage'].str.rstrip('%').astype(float)

# Display summary statistics for numerical columns to get an overview
summary_stats = cleaned_data.describe()

# Use IQR to identify outliers for one of the columns as an example, e.g., 'Rating'
Q1 = cleaned_data['Rating'].quantile(0.25)
Q3 = cleaned_data['Rating'].quantile(0.75)
IQR = Q3 - Q1

# Define outliers as those below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
outliers_rating = cleaned_data[(cleaned_data['Rating'] < (Q1 - 1.5 * IQR)) | (cleaned_data['Rating'] > (Q3 + 1.5 * IQR))]

summary_stats, outliers_rating


# In[ ]:





# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Selecting relevant features for predicting ratings
features = cleaned_data[['Memory', 'Storage', 'SellingPrice_EUR', 'OriginalPrice_EUR', 'DiscountPercentage']]
target = cleaned_data['Rating']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initializing and training the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predicting the ratings for the test set
y_pred = lr_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

rmse, r2


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculating the correlation matrix
correlation_matrix = cleaned_data[['Memory', 'Storage', 'SellingPrice_EUR', 'OriginalPrice_EUR', 'Rating']].corr()

# Plotting the heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()


# In[10]:


from sklearn.ensemble import GradientBoostingRegressor

# Initializing and training the Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predicting the ratings for the test set using Gradient Boosting
y_pred_gb = gb_model.predict(X_test)

# Evaluating the Gradient Boosting model
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

rmse_gb, r2_gb


# In[12]:


Statistical_analysis_phone_sales_data = '/Users/katie/Downloads/Statistical_analysis_phone_sales_data.csv'
cleaned_data.to_csv(Statistical_analysis_phone_sales_data, index=False)

Statistical_analysis_phone_sales_data

