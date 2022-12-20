#!/usr/bin/env python
# coding: utf-8

# 
# 
# <div class="markdown-google-sans">
# 
# </div>
# 
# 
# 
# <br/>
# Data : https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset?select=cs-training.csv<br/>
# 
# Context
# Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit.
# 
# Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.
# 
#  <br />

# ## 1. Overview Data Task
# 
# Question 1: Make the necessary steps to understand the data (suggested the following steps but no need to follow exactly) 
# - Observe the lines of the data
# - Total rows and columns 
# - Data type of each column 
# 
# ## 2. Clean Data Task
# - RevolvingUtilizationOfUnsecuredLines: 
#   + Def: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits
#   + Is there any missing value/outlier in this feature? 
#   + How should we deal with special values in this feature?
# 
# - DebtRatio: 
#   + Def: Monthly debt payments, alimony,living costs divided by monthy gross income
#   + Is there any missing value/outlier in this feature? 
#   + How should we deal with special values in this feature?
# 
# - MonthlyIncome: 
#   + Def: Monthly income
#   + Is there any missing value/outlier in this feature? 
#   + How should we deal with special values in this feature?
# 
# ## 3. Univariate Analysis Task
# - Perform univariate analysis to the features in the data and find the insights
# - Suggestions: 
#   + Start with the dependent variable, and then to the independent variables
#   + Find the distribution of the variables
# 
# ## 4. Bivariate Analysis Task
# - Perform bivariate analysis between the dependent variable and the independent variables 
# - Suggestions: 
#   + No suggestion for this part. Just go as wild as you can
# 

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns

sns.set(color_codes=True)


# In[2]:


df = pd.read_csv("C:/Users/ASUS/OneDrive/MĐ PYTHON/HW Clean Data/cs-training.csv")


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.rename(columns = {'Unnamed: 0':'ID'}, inplace = True)
df.head()


# 
# ## 3.1 Revolving Utilization Of Unsecured Lines

# In[8]:


# nó là tỷ lệ quay vòng ( so sánh giữa số tín dụng sử dụng và tổng hạn mức tín dụng)
# ==> nó chỉ có thể chạy từ 0-->1 


# In[9]:


df['RevolvingUtilizationOfUnsecuredLines'].describe()

# ==> Điều này là vô lý vì max chỉ có thể là 1


# In[10]:


df3=df[df['RevolvingUtilizationOfUnsecuredLines'] <=1]
sns.distplot(df3['RevolvingUtilizationOfUnsecuredLines'])


# In[11]:


len(df[(df['RevolvingUtilizationOfUnsecuredLines']>1)])


# this shows that about 3300 out of 150k observations got values more than 1 and hence it is not appropriate to consider all these as outliers and cap to 1. A better approach is to make these missing and impute the values

# In[12]:


# Fill Missing into the variables > 1
# ==> các giá trí lớn hơn 1 sẽ bị null 
def replace_1(x):
    if x > 1:
        return np.nan
    else:
        return x


# In[13]:


df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].apply(replace_1)
df['RevolvingUtilizationOfUnsecuredLines'].describe()


# In[14]:


#For imputation, we will use ffill method which will retain the distribution and mean of the variable.

df['RevolvingUtilizationOfUnsecuredLines'].fillna(method='ffill', inplace=True)

df['RevolvingUtilizationOfUnsecuredLines'].describe()


# # DebtRatio:
# 
# * Def: Monthly debt payments, alimony,living costs divided by monthy gross income
# * Is there any missing value/outlier in this feature?
# * How should we deal with special values in this feature?*

# ## 3.2 DebtRatio

# In[15]:


# the lower value of 0 is fine but max value is ridiculous as it is rarley more than 1

df['DebtRatio'].describe()


# In[16]:


df2 = df[df['DebtRatio']<=1]
sns.distplot(df2['DebtRatio'])


# In[17]:


df2=df[df['DebtRatio']>1]

df2['DebtRatio'].describe()


# Typical value of Debt Income ratio is 0.4. But almost 35000 observations got values higher than 1 and hence cannot be treated as outliers. Best approach right now is to leave this feature alone, later when we do the analysis and see that this feature has impact on the dependent variable, we will deal with the outliers. However, if you insist on dealing with the outliers right now, you can do the same as we did with RevolvingUtilizationOfUnsecuredLines

# ## 3.3 Monthly Income

# In[18]:


df['MonthlyIncome'].describe()


# There are missing values and Max value is too large. Min value of 0 is not ok as finance industry expect a minimum income of 1000.

# In[19]:


df['MonthlyIncome'].isnull().sum()


# In[20]:


len(df[df['MonthlyIncome']<1000])


# Number of obs below 1000 is too large. hence, it is not ok to treat it as outliers and cap to 1000. Lets treat it as missing and then impute the values.

# In[21]:


# Distribution shows that income smoothly decreases upto 25000 and then few outliers of huge values.

df2 = df[df['MonthlyIncome']<50000]
sns.distplot(df2['MonthlyIncome'].dropna())


# In[22]:


# Change all the income values > 25k to 25k

df.loc[df['MonthlyIncome']>25000, 'MonthlyIncome']=25000
df['MonthlyIncome'].describe()


# In[23]:


# Change all the income values < 1000 to NaN 

df.loc[df['MonthlyIncome']<1000, 'MonthlyIncome']=np.NaN
df['MonthlyIncome'].describe()


# In[24]:


# Fill the NaN values with mean of income values 

df['MonthlyIncome'].fillna(method='ffill', inplace=True)
df['MonthlyIncome'].describe()


# In[25]:


# The distribution makes much more sense now 

sns.distplot(df['MonthlyIncome'])


# In[26]:


# Finish Cleaning the data
df.loc[df['NumberRealEstateLoansOrLines']>5, 'NumberRealEstateLoansOrLines']=5
df.loc[df['NumberOfOpenCreditLinesAndLoans']>30, 'NumberOfOpenCreditLinesAndLoans']=30

df.loc[df['NumberOfDependents']>5, 'NumberOfDependents']=5
df['NumberOfDependents'].fillna(method='ffill', inplace=True)


# # Univariate Analysis Task
# * Perform univariate analysis to the features in the data and find the insights
# * Suggestions:
# - Start with the dependent variable, and then to the independent variables
# - Find the distribution of the variables

# In[27]:


df.head()


# In[28]:


#we will start with the dependent variable 

sns.countplot(x='SeriousDlqin2yrs', data=df)


# In[37]:


#About 6.7% customers were delinquents.

f,ax=plt.subplots(1,2,figsize=(14,6))

df['SeriousDlqin2yrs'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=False)
ax[0].set_title('SeriousDlqin2yrs')
ax[0].set_ylabel('')
# Vẽ pie chart by Series

sns.countplot('SeriousDlqin2yrs',data=df,ax=ax[1])
ax[1].set_title('SeriousDlqin2yrs')
plt.show();


# In[43]:


df['NumberOfDependents'].value_counts().plot(kind='bar', figsize = (10,10));


# In[53]:


df['NumberOfOpenCreditLinesAndLoans'].value_counts().plot(kind = 'bar')


# In[60]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df['NumberOfOpenCreditLinesAndLoans'],bins= 30)


# In[48]:


sns.distplot(df['age'])


# In[61]:


df2=df[df['DebtRatio']<=1]
sns.distplot(df2['DebtRatio'])


# ## 5.1 SeriousDlqin2yrs vs RevolvingUtilizationOfUnsecuredLines

# Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits

# In[62]:


df.groupby('SeriousDlqin2yrs')['RevolvingUtilizationOfUnsecuredLines'].agg(['count','mean'])


# In[76]:


df.groupby('SeriousDlqin2yrs')['RevolvingUtilizationOfUnsecuredLines'].mean().plot(kind='bar', color=['blue', 'green'])

#################################
####################
                                                ### Kiến thức mới ##
############################
#################################   


# In[65]:


df['RevolvingUtilizationOfUnsecuredLines'].describe()


# In[66]:


def cat_ruul(ruul):
    if ruul <0.03:
        return 1
    elif 0.03<= ruul <0.14:
        return 2
    elif 0.14<= ruul <0.52:
        return 3
    else:
        return 4


# In[67]:


df['ruul_cat'] = df['RevolvingUtilizationOfUnsecuredLines'].apply(cat_ruul)
df.head(3)


# In[68]:


df.groupby('ruul_cat')['RevolvingUtilizationOfUnsecuredLines'].agg(['min','max'])


# In[77]:


pd.crosstab(df.SeriousDlqin2yrs, df.ruul_cat, normalize='columns')


# In[79]:


sb=pd.crosstab(df.ruul_cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True);


# As expected, plot shows that there are more delinquents in the category of highest utilization. However there is not much difference between the fiirst two categories.

# ## 5.2 SeriousDlqin2yrs vs Age

# In[80]:


df.groupby('SeriousDlqin2yrs')['age'].agg(['count','mean'])


# In[81]:


df['age'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green'])


# Delinquent cusomters are younger than non-delinquent customers.

# In[83]:


df['age'].describe()


# In[84]:


def cat_ruul(ruul):
    if ruul <41:
        return 1
    elif 41<= ruul <52:
        return 2
    elif 52<= ruul <63:
        return 3
    else:
        return 4


# In[85]:


df['age_cat'] = df['age'].apply(cat_ruul)
df.head(3)


# In[86]:


# lets check if the categorization was done correctly
df.groupby('age_cat')['age'].agg(['min','max'])


# In[87]:


pd.crosstab(df.SeriousDlqin2yrs, df.age_cat, normalize='columns')


# In[88]:


sb=pd.crosstab(df.age_cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True)


# Plot shows that there are more proportion of delinquents in the category of youngest age. Delinquency decreases as age increases uniformly.

# In[89]:


df.age_cat.info


# In[ ]:




