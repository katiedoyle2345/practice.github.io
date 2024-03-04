#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
df = pd.read_csv('/Users/katie/Downloads/Sales.csv' ,delimiter=',',header=0,index_col=0)
df


# In[16]:


df.columns = df.columns.str.replace(' ', '')
df.columns


# In[17]:


df= df.dropna(axis=0, how='any')
df


# In[18]:


def convert_memory_to_float(memory_series):
    def convert_value(value):
        try:
            # Strip 'GB' and white spaces then convert to float
            return float(value.replace('GB', '').strip())
        except ValueError:
            # If conversion fails, return NaN
            return pd.NA
    
    return memory_series.apply(convert_value)
df = pd.DataFrame(df)
df['Memory'] = convert_memory_to_float(df['Memory'])
df


# In[21]:


df.dropna(axis=0, how='any', inplace=True)
df


# In[22]:


df['Storage'] = df['Storage'].str.extract('(\d+)').astype(float) 
df


# In[5]:


df['DiscountPercentage'] = ((df['OriginalPrice'] - df['SellingPrice']) / df['OriginalPrice']) * 100
df['DiscountPercentage'] = df['DiscountPercentage'].apply(lambda x: f"{x:.2f}%")
df.head()


# In[6]:


# Assuming the exchange rate is 1 INR = 0.011 EUR
exchange_rate_inr_to_eur = 0.011
df['SellingPrice_EUR'] = df['SellingPrice'] * exchange_rate_inr_to_eur
df['OriginalPrice_EUR'] = df['OriginalPrice'] * exchange_rate_inr_to_eur
df['SellingPrice_EUR'] = df['SellingPrice_EUR'].apply(lambda x: f"{x:.2f}€")
df['OriginalPrice_EUR'] = df['OriginalPrice_EUR'].apply(lambda x: f"{x:.2f}€")
df[['SellingPrice', 'SellingPrice_EUR', 'OriginalPrice', 'OriginalPrice_EUR']].head()


# In[8]:


df.drop(['SellingPrice', 'OriginalPrice'], axis=1, inplace=True)
df.head()


# In[10]:


cleaned_file_path = '/Users/katie/Downloads/Cleaned_Sales.csv'
df.to_csv(cleaned_file_path, index=False)

cleaned_file_path

