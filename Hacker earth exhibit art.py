#!/usr/bin/env python
# coding: utf-8

# # Problem Statement  
# To predict the cost required to ship the sculptures to customers based on the information provided in the dataset.

# # How will the company profit from this solution?
# For the transport company having these kind of solution will help in many ways.
# 1. They can prioritize transports based on how reputated the artist is and how big the client is, because they don't want to mess with good paying clients.
# 1. They can do batch transports of the sculptures whcih are intended to go at same nearby location which will make the delivery cost less.
# 1. The company can't keep track which factor is making the cost vary more.
# 1. The cost is overall cost that company had to incur in order to ship the product. It's not something they can know before hand. There is no specific formula calculator the cost.
# 1. That's why they want a Data Scientist to look into past data and try to build some models which will predict the cost for future shipments.
# 1. So that accordingly they can ask the customers price that is profitable to them. 
# 1. If model is not in place then there maybe a chance of the company asking lower price for shipment while the actual cost of shipment for delivery was much higher. Thus they have a high chance of making lots of losses

# ## Dataset Desciption  
# The dataset folder contains the following files:  
# train.csv: *6500 x 20*  
# test.csv: *3500 x 19*  

# ### Columns Provided in the Dataset  
# 1. Customer Id
# 1. Artist Name
# 1. Artist Reputation
# 1. Height
# 1. Width
# 1. Material
# 1. Price of Sculpture
# 1. Base Shipping Price
# 1. International
# 1. Express Shipment
# 1. Installation Included
# 1. Transport
# 1. Fragile
# 1. Customer Information
# 1. Remote Location
# 1. Scheduled Date
# 1. Delivery Date
# 1. Customer Location
# 1. Cost

# In[ ]:


# Importing Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Read the dataset
train_dataset = pd.read_csv('train (1).csv')
test_dataset = pd.read_csv('test.csv')


# ### Basic EDA

# # Identifying the number of features or columns

# ## Know all the names of the columns¶
# 

# In[ ]:


# get all column names
train_dataset.columns


# ## Knows more about the data in the columns like data type it contains and total samples of each

# In[ ]:


# Check which columns are having categorical, numerical or boolean values of train_dataset
train_dataset.info()


# In[ ]:


# Check which columns are having categorical, numerical or boolean values of test_dataset
test_dataset.info()


# 1. After checking the Dtypes of all the columns 
#     1. object - String values
#     1. float64 - Numerical values
# 1. There are more String values than the numerical values in the dataset

# # Know more mathematical relations of the dataset like count, min, max values, standarad deviation values, mean and different percentile values

# In[ ]:


# For train_dataset
# For more information on the dataset like the total count in all the columns
# min, max values and more information of the respective columns  
train_dataset.describe()


# In[ ]:


# for test_dataset
# For more information on the dataset like the total count in all the columns
# min, max values and more information of the respective columns
test_dataset.describe()


# ## Get the total number of samples in the dataset using the len() function

# In[ ]:


# len of train and test dataset
print("train data length :", len(train_dataset))
print("test data length :", len(test_dataset))


# ## Get unique values

# In[ ]:


# get how many unique values are in train_dataset
train_dataset.nunique()


# In[ ]:


# get how many unique values are in test_dataset
test_dataset.nunique()


# 
# 
# ## By the observation gather from the train_data.info(), we can know this columns are missing values in the train dataset
# 
#     1. Artist reputation
#     2. Height
#     3. Width
#     4. Weight
#     5. Transport
#     6. Material
#     7. Remote Location
# 
# ## By the observation gather from the test_data.info(), we can know this columns are missing values in the test dataset
# 
#     1. Artist Reputation
#     2. Height
#     3. Width
#     4. Weight
#     5. Transport
# 
# 

# ## Counting the total number of missing values¶
# 

# In[ ]:


# Check for missing values in all the columnns of the train_dataset
train_dataset.isnull().sum()


# In[ ]:


# Check for missing values in all the columnns of the test_dataset
test_dataset.isnull().sum()


# In[ ]:


# creating two empty list to store categorical column names and numerical column names respectively
categorical_list= []
var_list = []
# looping on whole dataset for geting list of categorical and numerical data column name and storing in respective list variable
for i in train_dataset.columns:
    if train_dataset[i].dtype=='object':
      categorical_list.append(i)
    elif train_dataset[i].dtype== 'float64':
      var_list.append(i)


# In[ ]:


print("Catergorical list",categorical_list)
print("numerical list",var_list)


# 
# ## Check for categorical columns in the dataset
# 
# #### By observing the train_data.info() cell, we can biforcate the datatype for which the object is the values which indicates those are the categorical columns. This dataset has more categorical columns than numerical values
# 
#     1. Customer Id
#     2. Artist Name
#     3. Material
#     4. International
#     5. Express Shipment
#     6. Installation included
#     7. Transport
#     8. Fragile
#     9. Customer Information
#     10. Remote Information
#     11. Scheduled Date
#     12. Delivery Date
#     13. Customer Location
# 
# 

# # Correlation Matrix
# 

# ## Why?
# #### A correlation matrix is a table showing correlation coefficients between variables.

# ### There are three broad reasons for computing a correlation matrix:
# 
#   1. To summarize a large amount of data where the goal is to see patterns. In our example above, the observable pattern is that all the variables highly correlate with each other.
#   2. To input into other analyses. For example, people commonly use correlation matrixes as inputs for exploratory factor analysis, confirmatory factor analysis, structural equation models, and linear regression when excluding missing values pairwise.
#   3. As a diagnostic when checking other analyses. For example, with linear regression, a high amount of correlations suggests that the linear regression estimates will be unreliable.

# In[ ]:


# Correlation metrix using pandas

corr = train_dataset[var_list].corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# ## From above correlation matrix:
# 1. Artist Reputation and all other numerical variables have almost no correlation.
# 
# 2. Height and Width are strongly correlated.
# 
# 3. Weight and Price of Sculpture are strongly correlated.
# 
# 4. Cost has weak correlation with Weight and Prie of Sculpture.

# In[ ]:


# Correlation metrix using seaborn
sns.heatmap(train_dataset[var_list].corr())
plt.show()


# # Chi-square Test

# 
# 
# 1. The Chi Square statistic is commonly used for testing relationships between categorical variables.
# 
# 2. The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.
# 
# 3. Example: Is there any significant relationship between gender and education qualification?
# 
# 4. The Chi-Square statistic is most commonly used to evaluate Tests of Independence when using a crosstabulation.
# 
# 5. Crosstabulation presents the distributions of two categorical variables simultaneously, with the intersections of the categories of the variables appearing in the cells of the table. that is values of one variable represents the row and other's value represents the column.
# 
# 6. Formula: x^2 = Summation of( (observed value - Expected value)^2/Expected value )
# 
# 7. The Chi-Square statistic is based on the difference between what is actually observed in the data and what would be expected if there was truly no relationship between the variables.
# 
# 8. This statistic can be evaluated by comparing the actual value against a critical value found in a Chi-Square distribution (where degrees of freedom is calculated as of rows – 1 x columns – 1), but it is easier to simply examine the p-value.
# 
# 9. To make a conclusion about the hypothesis with 95% confidence. Significance(p value of the Chi-square statistic) should be less than 0.05.
# 
#     1. Alpha level = 0.05(i.e 5%) 95% confidence about conclusion and 5% risk of not making a correct conclusion.
# 
#     2. Interpret the key results for Chi-Square Test for Association
# 
#         Determine whether the association between the variables is statistically significant.
# 
#         Examine the differences between expected counts and observed counts to determine which variable levels may have the most impact on association.
# 

# In[ ]:


# import necessary libraries for chi-square test

from scipy.stats import chi2_contingency
from scipy.stats import chi2
# creating function for performing chi-sqaure test on two columns
def perform_chi_square_test(var_1,var_2):
    
    #Contingency Table
    contigency_table = pd.crosstab(train_dataset[var_1],train_dataset[var_2])
    
    #Observed Values
    observed = contigency_table.values
    
    
    #Expected Values
    b = chi2_contingency(contigency_table)
    expected = b[3]
    
    #Degree of Freedom
    no_of_rows= len(contigency_table.iloc[0:,0])
    no_of_columns = len(contigency_table.iloc[0,0:])
    degree_f = (no_of_rows-1) * (no_of_columns-1)
    print("Degree of freedom:", degree_f)
    
    #Significance Level 5%
    alpha = 0.05
    print("Signifcance level:", alpha)
    
    #chi-square statistic
    chi_square = sum([(o-e)**2/e for o,e in zip(observed,expected)])
    chi_squared_stat = chi_square[0]+chi_square[1]
    print("chi square statics: ", chi_squared_stat)

    #critical_value
    critical_value = chi2.ppf(q= 1-alpha,df= degree_f)
    print("critical_value:", critical_value)
    #p-value
    p_value = 1- chi2.cdf(x=chi_squared_stat,df= degree_f)
    print("p_value :", p_value)


    # conditional statements for checking chi-sqaure test condition for hypothesis selection based on chi_square_statistic and critical_value     
    if chi_squared_stat >= critical_value:
      print("Retain H0, There is a relationship between 2 categorical varaibles")
    else:
      print("Retain H0, There is no relationship between 2 categorical varaibles")
    
    # conditional statements for checking chi-sqaure test condition for hypothesis selection based on p_value and alpha
    alpha = 0.05

    # interpret p-value

    if p_value <= alpha:
      print('Retain H0, There is a relationship between 2 categorical varaibles')
    else:
      print('Retain H0, There is no relationship between 2 categorical varaibles')


# In[ ]:


# looping on categorical data list and use function for performing chi-square test on columns from dataset
for x in categorical_list:
  for i in categorical_list:
    if i != x:
      print('chi-square test on:',x,'',i,'\n')
      perform_chi_square_test(x,i)
      print('----------------------------------\n')


# From above chi-square test:
# 
# - correlated variables:
# 
#   1. Customer Information and Transport 
#   2. Customer Information and Express Shipment
#   3. Fragile and Material 
#   4. Transport and Express Shipment
#   5. Transport and International 
# 

# # Variance inflation factor (VIF)

# 1. The variance inflation factor (VIF) quantifies the extent of correlation between one predictor and the other predictors in a model. 
# 2. It is used for diagnosing collinearity/multicollinearity. 
# 3. Higher values signify that it is difficult to impossible to assess accurately the contribution of predictors to a model.

# In[ ]:


# import statsmodle library for vif 
import statsmodels.api as sm


# In[ ]:


# creating a dataframe of just numerical values from train_dataset
train_for_vif = train_dataset[['Artist Reputation', 'Height', 'Width', 'Weight', 'Price Of Sculpture', 'Base Shipping Price']]
# target values from train_dataset
target = train_dataset['Cost']
# numerical values column names
names = ['Artist Reputation', 'Height', 'Width', 'Weight', 'Price Of Sculpture', 'Base Shipping Price']
# droping rows with from new dataframe empty cells
train_for_vif.dropna(inplace=True)
names


# In[ ]:


# loop for calculating VIF for each feature.

for i in range(0,len(names)):
  # taking one column as target variable
  y = train_for_vif.loc[:, train_for_vif.columns == names[i]]
  # taking all other remaining columns as fetaure variable
  x = train_for_vif.loc[:, train_for_vif.columns != names[i]]
  # Instantiating the statsmodel
  model = sm.OLS(y,x)
  results = model.fit()
  # fiting the OLS model on y and x
  rsq = results.rsquared
  # geting the r^2 value of results.
  vif = round(1/(1-rsq),2)
  # calculating vif value
  print("R Square value of {} columns is {} keeping all other columns as features".format(names[i],(round(rsq,2))))
  print("Variance inflaction factor of {} columns is {} \n".format(names[i],vif))
  
  


# Observations:
# 
# there is colinearity/multicolinearity between 
# variables as the VIF value is above 2.5
# 
# 1. There is high colinearity between height and all other variables.
# 2. There is high colinearity between Width and all other variables.
# 3. There is high colinearity between Price Of Sculpture and all other variables.

# ### ANOVA TEST

# ### Normality Assumption Check

# Before we perform the hypothesis test, we check if the assumptions for the one-way ANOVA hypothesis test are fulfilled. The samples are random and independent samples. Now, we check the normality assumption by plotting a normal probability plot (Q-Q plots) for each grouped variable.

# ## Homogeneity of variance Assumption Check

# ### Hypothesis Testing

# According to five steps process of hypothesis testing:
# H₀: μ₁= μ₂ = μ₃ = … = μ₆
# H₁: Not all salary means are equal
# α = 0.05
# According to F test statistics:

# Columns to perform anova test with:
# 1. Artist Name with all numerical columns
# 2. Material with all numerical columns
# 3. International with all numerical columns
# 4. Express Shipment with all numerical columns
# 5. Installation Included with all numerical columns
# 6. Transport with all numerical columns
# 7. Fragile with all numerical columns
# 8. Customer Information with all numerical columns
# 9. Remote Location with all numerical columns
# 10. Scheduled Date with all numerical columns
# 11. Delivery Date with all numerical columns
# 12. Customer Location with all numerical columns

# In[ ]:


import scipy.stats as stats


# In[ ]:


# function to perform anova test between two variables.
# perform anova test between two variables.

def perform_anova_test(x,y):
  # two variables of interest
  train_anova = train_dataset[[x, y]]
  groups = train_anova.groupby(x).count().reset_index()
  # groups.plot(kind='bar',x='major',y='salary')
  print(groups)

  # calculate ratio of the largest to the smallest sample standard deviation
  ratio = train_anova.groupby(x).std().max() / train_anova.groupby(x).std().min()
  print(ratio)


  # Create ANOVA backbone table
  data = [['Between Groups', '', '', '', '', '', ''], ['Within Groups', '', '', '', '', '', ''], ['Total', '', '', '', '', '', '']] 
  anova_table = pd.DataFrame(data, columns = ['Source of Variation', 'SS', 'df', 'MS', 'F', 'P-value', 'F crit']) 
  anova_table.set_index('Source of Variation', inplace = True)

  # calculate SSTR and update anova table
  x_bar = train_anova[y].mean()
  SSTR = train_anova.groupby(x).count() * (train_anova.groupby(x).mean() - x_bar)**2
  anova_table['SS']['Between Groups'] = SSTR[y].sum()

  # calculate SSE and update anova table
  SSE = (train_anova.groupby(x).count() - 1) * train_anova.groupby(x).std()**2
  anova_table['SS']['Within Groups'] = SSE[y].sum()

  # calculate SSTR and update anova table
  SSTR = SSTR[y].sum() + SSE[y].sum()
  anova_table['SS']['Total'] = SSTR

  # update degree of freedom
  anova_table['df']['Between Groups'] = train_anova[x].nunique() - 1
  anova_table['df']['Within Groups'] = train_anova.shape[0] - train_anova[x].nunique()
  anova_table['df']['Total'] = train_anova.shape[0] - 1

  # calculate MS
  anova_table['MS'] = anova_table['SS'] / anova_table['df']

  # calculate F 
  F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
  anova_table['F']['Between Groups'] = F

  # p-value
  anova_table['P-value']['Between Groups'] = 1 - stats.f.cdf(F, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

  # F critical 
  alpha = 0.05
  # possible types "right-tailed, left-tailed, two-tailed"
  tail_hypothesis_type = "two-tailed"
  if tail_hypothesis_type == "two-tailed":
      alpha /= 2
  anova_table['F crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

  # Final ANOVA Table
  print(anova_table)


  # The p-value approach
  print("Approach 1: The p-value approach to hypothesis testing in the decision rule")
  conclusion = "Failed to reject the null hypothesis."
  if anova_table['P-value']['Between Groups'] <= alpha:
      conclusion = "Null Hypothesis is rejected."
  print("F-score is:", anova_table['F']['Between Groups'], " and p value is:", anova_table['P-value']['Between Groups'])    
  print(conclusion)
      
  # The critical value approach
  print("\n--------------------------------------------------------------------------------------")
  print("Approach 2: The critical value approach to hypothesis testing in the decision rule")
  conclusion = "Failed to reject the null hypothesis."
  if anova_table['F']['Between Groups'] > anova_table['F crit']['Between Groups']:
      conclusion = "Null Hypothesis is rejected."
  print("F-score is:", anova_table['F']['Between Groups'], " and critical value is:", anova_table['F crit']['Between Groups'])
  print(conclusion)


# ### Anova test between Artist Name and all numerical columns

# In[ ]:


# perform_anova_test Artist Name and Artist Reputation
perform_anova_test('Artist Name','Artist Reputation')
# perform_anova_test Artist Name and Height
perform_anova_test('Artist Name','Height')
# perform_anova_test Artist Name and Width
perform_anova_test('Artist Name', 'Width')
# perform_anova_test Artist Name and Weight
perform_anova_test('Artist Name', 'Weight')
# perform_anova_test Artist Name and Price Of Sculptureprint
perform_anova_test('Artist Name', 'Price Of Sculpture')
# perform_anova_test Artist Name and Base Shipping Price
perform_anova_test('Artist Name', 'Base Shipping Price')
# perform_anova_test Artist Name and Cost
perform_anova_test('Artist Name', 'Cost')


# ### Anova test between Material and all numerical columns

# In[ ]:


# perform_anova_test Material and Artist Reputation
perform_anova_test('Material', 'Artist Reputation')
# perform_anova_test Material and Height
perform_anova_test('Material', 'Height')
# perform_anova_test Material and Width
perform_anova_test('Material', 'Width')
# perform_anova_test Material and Weight
perform_anova_test('Material', 'Weight')
# perform_anova_test Material and Price Of Sculpture
perform_anova_test('Material', 'Price Of Sculpture')
# perform_anova_test Material and Base Shipping Price
perform_anova_test('Material', 'Base Shipping Price')
# perform_anova_test Material and Cost
perform_anova_test('Material', 'Cost')


# ### Anova test between International and all numerical columns

# In[ ]:


# perform_anova_test International and Artist Reputation
perform_anova_test('International', 'Artist Reputation')
# perform_anova_test International and Height
perform_anova_test('International', 'Height')
# perform_anova_test International and Width
perform_anova_test('International', 'Width')
# perform_anova_test International and Weight
perform_anova_test('International', 'Weight')
# perform_anova_test International and Price Of Sculpture
perform_anova_test('International', 'Price Of Sculpture')
# perform_anova_test International and Base Shipping Price
perform_anova_test('International', 'Base Shipping Price')
# perform_anova_test International and Cost
perform_anova_test('International', 'Cost')


# ### Anova test between Fragile and all numerical columns

# In[ ]:


# perform_anova_test Fragile and Height
perform_anova_test('Fragile','Height')
# perform_anova_test Fragile and Width
perform_anova_test('Fragile','Width')
# perform_anova_test Fragile and Weight
perform_anova_test('Fragile','Weight')
# perform_anova_test Fragile and Price Of Sculpture
perform_anova_test('Fragile','Price Of Sculpture')
# perform_anova_test Fragile and Base Shipping Price
perform_anova_test('Fragile','Base Shipping Price')
# perform_anova_test Fragile and 
perform_anova_test('Fragile','Cost')


# ### Anova test between Customer Information and all numerical columns

# In[ ]:


train_dataset.columns


# In[ ]:


# perform_anova_test Customer Information and Artist Reputation
perform_anova_test('Customer Information','Artist Reputation')
# perform_anova_test Customer Information and Height
perform_anova_test('Customer Information','Height')
# perform_anova_test Customer Information and Width
perform_anova_test('Customer Information','Width')
# perform_anova_test Customer Information and Weight
perform_anova_test('Customer Information','Weight')
# perform_anova_test Customer Information and Price Of Sculpture
perform_anova_test('Customer Information','Price Of Sculpture')
# perform_anova_test Customer Information and Base Shipping Price
perform_anova_test('Customer Information','Base Shipping Price')
# perform_anova_test Customer Information and Cost
perform_anova_test('Customer Information','Cost')


# ### Anova test between Remote Locaation and all numerical columns

# In[ ]:


# perform_anova_test Remote Location and Artist Reputation
perform_anova_test('Remote Location','Artist Reputation')
# perform_anova_test Remote Location and Height
perform_anova_test('Remote Location','Height')
# perform_anova_test Remote Location and Width
perform_anova_test('Remote Location','Width')
# perform_anova_test Remote Location and Weight
perform_anova_test('Remote Location','Weight')
# perform_anova_test Remote Location and Price Of Sculpture
perform_anova_test('Remote Location','Price Of Sculpture')
# perform_anova_test Remote Location and Base Shipping Price
perform_anova_test('Remote Location','Base Shipping Price')
# perform_anova_test Remote Location and Cost
perform_anova_test('Remote Location','Cost')


# ### Anova test between Scheduled Date and all numerical columns

# In[ ]:


train_dataset.columns


# In[ ]:


# perform_anova_test Scheduled Date and Artist Reputation
perform_anova_test('Remote Location','Artist Reputation')
# perform_anova_test Scheduled Date and Height
perform_anova_test('Remote Location','Height')
# perform_anova_test Scheduled Date and Width
perform_anova_test('Remote Location','Width')
# perform_anova_test Scheduled Date and Weight
perform_anova_test('Remote Location','Weight')
# perform_anova_test Scheduled Date and Price Of Sculpture
perform_anova_test('Remote Location','Price Of Sculpture')
# perform_anova_test Scheduled Date and Base Shipping Price
perform_anova_test('Remote Location','Base Shipping Price')
# perform_anova_test Scheduled Date and Cost
perform_anova_test('Remote Location','Cost')


# In[ ]:





# ### Anova test between Delivery Date and all numerical columns

# In[ ]:


# perform_anova_test Delivery Date and Artist Reputation
perform_anova_test('Delivery Date','Artist Reputation')
# perform_anova_test Delivery Date and Height
perform_anova_test('Delivery Date','Height')
# perform_anova_test Delivery Date and Width
perform_anova_test('Delivery Date','Width')
# perform_anova_test Delivery Date and Weight
perform_anova_test('Delivery Date','Weight')
# perform_anova_test Delivery Date and Price Of Sculpture
perform_anova_test('Delivery Date','Price Of Sculpture')
# perform_anova_test Delivery Date and Base Shipping Price
perform_anova_test('Delivery Date','Base Shipping Price')
# perform_anova_test Delivery Date and Cost
perform_anova_test('Delivery Date','Cost')


# ### Anova test between Customer Location and all numerical columns

# In[ ]:


# perform_anova_test Customer Location and Artist Reputation
perform_anova_test('Customer Information','Artist Reputation')
# perform_anova_test Customer Location and Height
perform_anova_test('Customer Information','Height')
# perform_anova_test Customer Location and Width
perform_anova_test('Customer Information','Width')
# perform_anova_test Customer Location and Weight
perform_anova_test('Customer Information','Weight')
# perform_anova_test Customer Location and Price Of Sculpture
perform_anova_test('Customer Information','Price Of Sculpture')
# perform_anova_test Customer Location and Base Shipping Price
perform_anova_test('Customer Information','Base Shipping Price')
# perform_anova_test Customer Location and Cost
perform_anova_test('Customer Information','Cost')


# # Scatter Plot

# 
# 
# 1. A scatter plot is a type of plot using Cartesian coordinates to display values for typically two variables for a set of data.
# 
# 2. The data are displayed as a collection of points, each having the value of one variable determining the position on the horizontal axis and the value of the other variable determining the position on the vertical axis.
# 
# 3. Scatter plot's are used to observe and show relationships between two numeric variables.
# 

# In[ ]:


# Scatter plot using matplotlib 
# create function for ploting scatterplot between two columns of dataset
def plot_scatter(x,y):
  plt.figure()
  plt.xlabel(x)
  plt.ylabel(y)
  plt.scatter(train_dataset[x],train_dataset[y])
  plt.show()


# Loop through numerical data list and use function to scatter plot between two columns
for i in var_list:
  for j in var_list:
    if i != j:
      plot_scatter(i,j)

# plot all pair of scatterplot by writing code for each pair


# From above scatter plot
# 
# 1. Increase in value on Width axis results in increase of values on Height axis. That is they are positively correlated.
# 
# 2. Increase in value on Weight axis results in increase of values on Price Of Sculpture and Cost axis. Weight and Price Of Sculpture are strongly correlated , Weight and Cost are weakly correlated.

# # Histogram

# 
# 
# 1. A histogram is an approximate representation of the distribution of numerical data.
# 
# 2. To construct a histogram, the first step is to "bin" (or "bucket") the range of values—that is, divide the entire range of values into a series of intervals—and then count how many values fall into each interval.
# 
# 3. The words used to describe the patterns in a histogram are: "symmetric", "skewed left" or "right", "unimodal", "bimodal" or "multimodal".
# 

# In[ ]:


# Histogram using pandas 
plt.figure(figsize=(16,12))
train_dataset[var_list].hist()


# From the above histogram
# 
# 1. Artist Reputation data distribution is symmetric.
# 
# 2. Height data distribution is skewed left.
# 
# 3. Width data distribution is skewed left.
# 
# 4. Weight data distribution is skewed left.
# 
# 5. Price Of Sculpture data distribution is skewed left.
# 
# 6. Base Shipping Price data distribution is skewed left.
# 
# 3. Cost data distribution is skewed left.
# 

# ## groupby
# 
# You can use groupby to chunk up your data into subsets for further analysis.

# In[ ]:


# group data by Artist Reputation and plot count plot
train_dataset.groupby('Artist Reputation').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each value of Artist Reputation is distributed unequally.
# 2. But for overall data distribution of Artist Reputation is symmetrical

# In[ ]:


# group data by Height and plot count plot
train_dataset.groupby('Height').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each value of Height is distributed unequally.
# 2. But for overall data distribution of Height is skewed left.

# In[ ]:


# group data by Width and plot count plot
train_dataset.groupby('Width').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each value of Width is distributed unequally.
# 2. But for overall data distribution of Width is skewed left.

# In[ ]:


# group data by Weight and plot count plot
train_dataset.groupby('Weight').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each value of Weight is distributed unequally.
# 2. But for overall data distribution of Weight is skewed left.

# In[ ]:


# group data by Price Of Sculpture and plot count plot
train_dataset.groupby('Price Of Sculpture').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each value of Price Of Sculpture is distributed unequally.
# 2. But for overall data distribution of Price Of Sculpture is skewed left.

# In[ ]:


# group data by Base Shipping Price and plot count plot
train_dataset.groupby('Base Shipping Price').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each Base Shipping Price value is distributed unequally.
# 2. Base Shipping Price is skewed left.

# In[ ]:


# group data by International and plot count plot
train_dataset.groupby('International').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. Number of varibales having "No" as value of international is almost double the number of varibales having "Yes" value.

# In[ ]:


# group data by Express Shipment and plot count plot
train_dataset.groupby('Express Shipment').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. Number of variables having "No" as value Express Shipment is almost double the number of variables having "Yes" value.

# In[ ]:


# group data by Installation Included and plot count plot
train_dataset.groupby('Installation Included').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. Number of variables having "No" as value international is almost double the number of variables having "Yes" value.

# In[ ]:


# group data by Transport and plot count plot
train_dataset.groupby('Transport').count().plot(kind='bar', figsize=(20,6))


# 1. Number of variables having value as Roadways is greater than Number of variables having value Airways
# 2. Number of variables having value Waterways is half as compared to those of Airways and Roadways

# In[ ]:


# group data by Fragile and plot count plot
train_dataset.groupby('Fragile').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. 10% of the Sculptures are Fragile

# In[ ]:


# group data by Customer Information and plot count plot
train_dataset.groupby('Customer Information').count().plot(kind='bar', figsize=(20,6))


# from above graph
# 
# Most of the customers blongs to Working class.

# In[ ]:


# group data by Remote Location and plot count plot
train_dataset.groupby('Remote Location').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. Number sample having remote location is double the number of sample which are not.

# In[ ]:


# group data by Scheduled Date and plot count plot
train_dataset.groupby('Scheduled Date').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each Scheduled Date is distributed  unequally.

# In[ ]:


# group data by Delivery Date and plot count plot
train_dataset.groupby('Delivery Date').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each Delivery Date is distributed unequally.

# In[ ]:


# group data by Customer Location and plot count plot
train_dataset.groupby('Customer Location').count().plot(kind='bar', figsize=(20,6))


# from above graph:
# 1. All data for each customer locaton value is distributed equally.

# In[ ]:


# group data by Cost and plot count plot
train_dataset.groupby('Cost').count().plot(kind='bar', figsize=(20,6))


# # Box Plot

# A boxplot is a standardized way of displaying the dataset based on a five-number summary:
# 
#     1. Minimum (Q0 or 0th percentile): the lowest data point excluding any outliers.
# 
#     2. Maximum (Q4 or 100th percentile): the largest data point excluding any outliers.
# 
#     3. Median (Q2 or 50th percentile): the middle value of the dataset.
# 
#     4. First quartile (Q1 or 25th percentile): also known as the lower quartile qn(0.25), is the median of the lower half of the dataset.
# 
#     5. Third quartile (Q3 or 75th percentile): also known as the upper quartile qn(0.75), is the median of the upper half of the dataset
# 

# In[ ]:


# box plot using pandas 
# box plot for Artist Reputation column
train_dataset.boxplot(column = 'Artist Reputation',figsize=(12,10))


# from above box plot graph:
# 
# - Artist reputation
#   1. 25% of Artist reputation have value between range 0 to 0.24.
#   2. 25% of Artist reputation have value between range 0.24 to 0.45.
#   3. 25% of Artist reputation have value between range 0.45 to 0.68.
#   4. 25% of Artist reputation have value between range 0.68 to 1.
# 
# 
# - The mean Artist Reputation is around 0.45

# In[ ]:


# box plot using pandas 
# box plot for Height column
train_dataset.boxplot(column = 'Height',figsize=(12,10))


# from above box plot graph:
# 
# - Height
#   1. 25% of Heigth have value between range 3 to 12.
#   2. 25% of Artist reputation have value between range 12 to 20.
#   3. 25% of Artist reputation have value between range 20 to 30.
#   4. 25% of Artist reputation have value between range 30 to 73.
# 
# 
# - The mean Height is around 20.

# In[ ]:


# box plot using pandas 
# box plot for Width column 
train_dataset.boxplot(column = 'Width',figsize=(12,10))


# from above box plot graph:
# 
# - Width
#   1. less than 25% of Width have value between range 2 to 6.
#   2. less 50% of Width have value between range 2 to 8.
#   3. less than 75% of Width have value between range 2 to 12.
#   4. More than 25% Width have value between range 12 to 22.
#   5. very few values of width are in range between 22 to 50. that is these are outliers
# 
# 
# - The mean Width is around 10

# In[ ]:


# box plot using pandas 
# box plot for Weight column 
train_dataset.boxplot(column = 'Weight',figsize=(12,10))


# from above box plot graph:
# 
# - Height
#   1. less than Height of Width have value between range 2 to 6.
#   2. less 50% of Height have value between range 2 to 8.
#   3. less than 75% of Height have value between range 2 to 12.
#   4. More than 25% Height have value between range 12 to 22.
#   5. very few values of Height are in range between 22 to 50. that is these are outliers
# 
# 
# - The mean Height is around 10

# In[ ]:


# box plot using pandas 
# box plot for Cost column 
train_dataset.boxplot(column = 'Cost',figsize=(12,10))


# from above box plot graph:
# 
# - Cost
#   1. Most of the Cost value is around 0
#   2. very few values are above 0.2. that is they are outliers

# In[ ]:


# box plot using seaborn 

sns.boxplot(x='Artist Reputation',  data= train_dataset)
plt.show()
sns.boxplot(x='Height',  data= train_dataset)
plt.show()
sns.boxplot(x='Width',  data= train_dataset)
plt.show()
sns.boxplot(x='Weight',  data= train_dataset)
plt.show()
sns.boxplot(x='Price Of Sculpture',  data= train_dataset)
plt.show()
sns.boxplot(x='Base Shipping Price',  data= train_dataset)
plt.show()
sns.boxplot(x='Cost',  data= train_dataset)
plt.show()


# 
# # Violin Plot
# 
# 
# 
# 1. A violin plot is a method of plotting numeric data.
# 
# 1. Violin plots are similar to box plots, except that they also show the probability density of the data at different values, usually smoothed by a kernel density estimator.
# 
# 3. It has:
# 
#     1. Median (a white dot on the violin plot)
#     2. Interquartile range (the black bar in the center of violin)
#     3. The lower/upper adjacent values (the black lines stretched from the bar) — defined as first quartile — 1.5 IQR and third quartile + 1.5 IQR respectively.

# In[ ]:


# violin plot for Height and Width columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Height', y='Width', data=train_dataset, palette='rainbow')


# from above violin plot:
# 1. The distribution between lower adjacent value and upper adjacent value is almost symmetrical.
# 2. also there is higher observation probability at the between first quartile and third quartile.
# 3. The Width range is increasing as we move right on the axis of Height

# In[ ]:


# violin plot for Height and Artist Reputation columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Height', y='Artist Reputation', data=train_dataset, palette='rainbow')


# from above violin plot:
# 1. The distribution between lower adjacent value and upper adjacent value is almost symmetrical.
# 2. also there is higher observation probability at the between first quartile and third quartile.

# In[ ]:


# violin plot for Height and Weight columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Height', y='Weight', data=train_dataset, palette='rainbow')


# from above violin plot:
# 1. The distribution between lower adjacent value and upper adjacent value is almost symmetrical.
# 2. also there is higher observation probability at the between first quartile and third quartile.
# 3. Weight distribution for all values of height is simmilar.
# 4. The Weight range is increasing and then decreases as we move right on the axis of Height

# In[ ]:


# violin plot for Height and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Height', y='Price Of Sculpture', data=train_dataset, palette='rainbow')


# from above violin plot:
# 1. The distribution between lower adjacent value and upper adjacent value is almost symmetrical.
# 2. also there is higher observation probability at the between first quartile and third quartile.
# 3. Price Of Sculpture distribution for all values of height is simmilar.
# 4. The Price Of Sculpture range is increasing and then decreases as we move right on the axis of Height
# 5. There is very weak correlataion between Price Of Sculpture and Height.

# In[ ]:


# violin plot for Height and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Height', y='Base Shipping Price', data=train_dataset, palette='rainbow')


# from above violin plot:
# 1. The distribution between lower adjacent value and upper adjacent value is unsymmetrical.
# 
# 2. Also there is higher observation probability at the between first quartile and third quartile.
# 
# 3. Base Shipping Price distribution for all values of height is not simmilar.
# 
# 4. The Base Shipping Price range is increasing and then decreases as we move right on the axis of Height

# In[ ]:


# violin plot for Height and Cost columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Height', y='Cost', data=train_dataset, palette='rainbow')


# from above violin plot:
# 1. The distribution between lower adjacent value and upper adjacent value is almost symmetrical.
# 2. also there is higher observation probability at the between first quartile and third quartile.
# 3. Cost distribution for all values of height is simmilar.
# 4. The Cost range is increasing and then decreases as we move right on the axis of Height

# In[ ]:


# violin plot for Artist Reputation and Width columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Artist Reputation', y='Width', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# 1. Width distribution for all values of Artist Reputation is simmilar.
# 
# 2. The Width and Artist Reputation have no correlation

# In[ ]:


# violin plot for Artist Reputation and Weight columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Artist Reputation', y='Weight', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# 1. Weight distribution for all values of Artist Reputation is simmilar.
# 
# 2. The Weight and Artist Reputation have no correlation

# In[ ]:


# violin plot for Artist Reputation and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Artist Reputation', y='Price Of Sculpture', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# 1. Price Of Sculpture distribution for all values of Artist Reputation is simmilar.
# 
# 2. The Price Of Sculpture and Artist Reputation have no correlation

# In[ ]:


# violin plot for Artist Reputation and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Artist Reputation', y='Base Shipping Price', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# 1. Base Shipping Price distribution for all values of Artist Reputation is simmilar.
# 
# 2. The Base Shipping Price and Artist Reputation have no correlation

# In[ ]:


# violin plot for Artist Reputation and Cost columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Artist Reputation', y='Price Of Sculpture', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# 1. Cost distribution for all values of Artist Reputation is simmilar.
# 
# 2. The Cost and Artist Reputation have no correlation

# In[ ]:


# violin plot for Width and Weight columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Width', y='Weight', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# The Weight and Width have very weak correlation

# In[ ]:


# violin plot for Width and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Width', y='Price Of Sculpture', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# The Price Of Sculpture and Width have very weak correlation

# In[ ]:


# violin plot for Width and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Width', y='Base Shipping Price', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# The Base shipping price and Width are weakly correlated

# In[ ]:


# violin plot for Width and Cost columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Width', y='Cost', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# The Cost and Width have no correlation

# In[ ]:


# violin plot for Price Of Sculpture and Weight columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Price Of Sculpture', y='Weight', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# Values of cost is almost around zero for weight at the begining but starts to show high change in value for larger values of price of sculpture  

# In[ ]:


# violin plot for Price Of Sculpture and Cost columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Price Of Sculpture', y='Cost', data=train_dataset, palette='rainbow')


# from above violine plot:
# the highest priced scuplture have larger weight than low price sculpture.

# In[ ]:


# violin plot for Weight and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Weight', y='Base Shipping Price', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# The Base Shipping and Weight have very weak correlation

# In[ ]:


# violin plot for Weight and Cost columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Weight', y='Cost', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# Values of cost is almost around zero for weight at the begining but starts to show high change in value for larger values of weight  

# In[ ]:


# violin plot for Base Shipping Price and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Base Shipping Price',y='Price Of Sculpture', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# The Base Shipping and Price of Sculpture have no correlation

# In[ ]:


# violin plot for Price Of Sculpture and Cost columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Price Of Sculpture', y='Cost', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# Values of cost is almost around zero for price of sculpture at the begining but starts to show high change in value for larger values of price of sculpture

# In[ ]:


# violin plot for Base Shipping Price and Cost columns
plt.figure(figsize=(20,6))
sns.violinplot(x='Base Shipping Price',y='Cost', data=train_dataset, palette='rainbow')


# from above violin plot:
# 
# Values of cost is almost around zero for Base Shipping Price at the begining but starts to show high change in value for larger values of Base Shipping Price.

# 
# # Boxenplot
# 
# 1. The boxen plot, otherwise known as a Letter-value plot, is a box plot meant for large data sets (n > 10,000).
# 
# 2. The Boxen plot is very similar to box plot, except for the fact that it plots different quartile values.
# 
# 3. By plotting different quartile values, we are able to understand the shape of the distribution particularly in the head end and tail end.
# 

# In[ ]:


# Boxen plot for Artist Reputation and Height columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Artist Reputation', y = 'Height', 
              data = train_dataset)


# from above boxen plot:
# 1. The distribution between lower adjacent value and upper adjacent value is symmetrical.
# 2. There is no relation between height and Artist Reputation

# In[ ]:


# Boxen plot for Artist Reputation and Width columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Artist Reputation', y = 'Width', 
              data = train_dataset)


# from above boxen plot:
# 1. The distribution between lower adjacent value and upper adjacent value is symmetrical.
# 2. There is no relation between Width and Artist Reputation

# In[ ]:


# Boxen plot for Artist Reputation and Weight columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Artist Reputation', y = 'Weight', 
              data = train_dataset)


# from above boxen plot:
# 
# values of weight are between 0.0 to 0.2 for most values of Artist reputation but there are some values above 0.2

# In[ ]:


# Boxen plot for Artist Reputation and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Artist Reputation', y = 'Price Of Sculpture', 
              data = train_dataset)


# from above boxen plot:
# 
# values of price of sculpture are between 0.0 to 50000 for most values of Artist reputation but there are some values above 50000

# In[ ]:


# Boxen plot for Artist Reputation and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Artist Reputation', y = 'Base Shipping Price', 
              data = train_dataset)


# from above boxen plot:
# 1. The distribution between lower adjacent value and upper adjacent value is symmetrical.
# 2. There is no rcorelation between Base Shipping Price and Artist Reputation

# In[ ]:


# Boxen plot for Artist Reputation and Cost columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Artist Reputation', y = 'Cost', 
              data = train_dataset)


# from above boxen plot:
# 
# values of cost are between 0.0 to 0.1 for most values of Artist reputation but there are some values above 0.1

# In[ ]:


# Boxen plot for Height and Width columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Height', y = 'Width',
              data = train_dataset)


# from above boxen plot:
# 1. There is a positive correlation between height and Width.

# In[ ]:


# Boxen plot for Height and Weight columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Height', y = 'Weight',
              data = train_dataset)


# from above boxen plot:
# 
# range of values of weight increases with increase in height values and then decreases with decrease in height value.
# 
# values of weight are between 0.0 to 0.2 for most values of Height but there are some values above 0.2

# In[ ]:


# Boxen plot for Height and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Height', y = 'Price Of Sculpture',
              data = train_dataset)


# from above boxen plot:
# 
# range of values of weight price of sculpture with increase in height values and then decreases with decrease in height value.
# 
# values of weight are between 0.0 to 50000 for most values of Height but there are some values above 50000

# In[ ]:


# Boxen plot for Height and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Height', y = 'Base Shipping Price',
              data = train_dataset)


# from above boxen plot:
# 
# 1. There is weak correlation between height and Base Shipping Price

# In[ ]:


# Boxen plot for Height and Cost columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Height', y = 'Cost',
              data = train_dataset)


# In[ ]:


# Boxen plot for Width and Weight columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Width', y= 'Weight',
              data = train_dataset)


# from above boxen plot:
# 
# 1. There is no relation between Width and Weight

# In[ ]:


# Boxen plot for Width and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Width', y= 'Price Of Sculpture',
              data = train_dataset)


# from above boxen plot:
# 
# 1. There is very weak correlation between Width and Price Of Sculpture

# In[ ]:


# Boxen plot for Width and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Width', y= 'Base Shipping Price',
              data = train_dataset)


# from above boxen plot:
# 
# 1. There is weak correlation between Width and Base Shipping Price

# In[ ]:


# Boxen plot for Width and Cost columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Width', y= 'Cost',
              data = train_dataset)


# from above boxen plot:
# 
# 1. There is weak correlation between Width and cost Price also there are many outliers

# In[ ]:


# Boxen plot for Weight and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Weight', y ='Price Of Sculpture'
              data = train_dataset)


# In[ ]:


# Boxen plot for Weight and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Weight', y ='Base Shippinh Price'
              data = train_dataset)


# from above boxen plot:
# 
# 1. There is very weak correlation between Weight and Base Shipping Price

# In[ ]:


# Boxen plot for Weight and Cost columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Weight', y ='Cost'
              data = train_dataset)


# from above boxen plot:
# 
# values of cost are between 0.0 to 0.1 for most values of Weight but there are some values above 0.1

# In[ ]:


# Boxen plot for Price Of Sculpture and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Price Of Sculpture',y = 'Base Shipping Price'
              data = train_dataset)


# from above boxen plot:
# 
# 1. There is very weak correlation between Price Of Sculpture and Base Shipping Price

# In[ ]:


# Boxen plot for Price Of Sculpture and Cost columns
plt.figure(figsize=(20,6))
sns.boxenplot(x ='Price Of Sculpture',y = 'Cost'
              data = train_dataset)


# from above boxen plot:
# 
# Values of cost is almost around zero for price of sculpture at the begining but starts to show high change in value for larger values of price of sculpture

# In[ ]:


# Boxen plot for Base Shipping Price and Cost columns
plt.figure(figsize=(20,6))
sns.boxenplot(x = 'Base Shipping Price'. y= 'Cost',
              data = train_dataset)


# from above boxen plot:
# 
# Values of cost is almost around zero for Base Shipping Price at the begining but starts to show high change in value for larger values of Base Shipping Price.

# 
# # Point Plot
# 
# 
# 1. A point plot uses scatter plot glyphs to visualize features like point estimates and confidence intervals.
# 
# 2. A point plot uses scatter plot points to represent the central tendency of numeric data.
# 
# 3. These plots make use of error bars to indicate any uncertainty around the numeric

# In[ ]:


# point plot for Artist Reputation and Height columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Artist Reputation',y='Height', data = dataset, join=False,palette='rainbow')


# From above point plot
# 
# The error bar shows that there is variability of association with each Y and X center point value. that is the Height has standard deviations of vales for each Artist Reputation value

# In[ ]:


# point plot for Artist Reputation and Width columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Artist Reputation',y='Width', data = dataset, join=False,palette='rainbow')


# From above point plot
# 
# The error bar shows that there is variability of association with each Y and X center point value. that is the Width has standard deviations of vales for each Artist Reputation value

# In[ ]:


# point plot for Artist Reputation and Weight columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Artist Reputation',y='Weight', data = dataset, join=False,palette='rainbow')


# From above point plot
# 
# The error bar shows that there is variability of association with each Y and X center point value. that is the Weight has standard deviations of vales for each Artist Reputation value

# In[ ]:


# point plot for Artist Reputation and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Artist Reputation',y='Price Of Sculpture', data = dataset, join=False,palette='rainbow')


# In[ ]:


# point plot for Artist Reputation and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Artist Reputation',y='Base Shipping Price', data = dataset, join=False,palette='rainbow')


# In[ ]:


# point plot for Artist Reputation and  Cost columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Artist Reputation',y='Cost', data = dataset, join=False,palette='rainbow')


# From above pointplot
# 
# 1. Most of the points are around 0.
# 2. Very few points above 0.

# In[ ]:


# point plot for Height and Width columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Height', y='Width', data = dataset, join=False,palette='rainbow')


# From above point plot
# 
# There is a increase in width when there is a increase in Height.
# That is both are correlated

# In[ ]:


# point plot for Height and Weight columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Height',y= 'Weight' data = dataset, join=False,palette='rainbow')


# From above pointplot
# 
# 1. Most of the points are around 0.
# 2. Very few points above 0.

# In[ ]:


# point plot for Height and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Height',y= 'Price Of Sculpture', data = dataset, join=False,palette='rainbow')


# From above pointplot
# 
# 1. Most of the points are around 0.
# 2. Very few points above 0.

# In[ ]:


# point plot for Height and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Height',y= 'Base Shipping Price', data = dataset, join=False,palette='rainbow')


# From above point plot
# 
# There is a increase in Base Shipping Price when there is a increase in Height. That is they are correlated

# In[ ]:


# point plot for Height and Cost columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Height',y= 'Cost', data = dataset, join=False,palette='rainbow')


# From above pointplot
# 
# 1. Most of the points are around 0.
# 2. Very few points above 0.

# In[ ]:


# point plot for Width and Weight columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Width',y= 'Weight', data = dataset, join=False,palette='rainbow')


# From above pointplot
# 
# 1. Most of the points are around 0.
# 2. Very few points above 0.

# In[ ]:


# point plot for Width and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Width',y= 'Price Of Sculpture', data = dataset, join=False,palette='rainbow')


# From above pointplot
# 
# 1. Most of the points are around 0.
# 2. Very few points above 0.

# In[ ]:


# point plot for Width and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Width',y= 'Base Shipping Price', data = dataset, join=False,palette='rainbow')


# From above point plot
# 
# There is a increase in Base Shipping Price when there is a increase in Widtg. That is both are correlated

# In[ ]:


# point plot for Width and Cost columns
plt.figure(figsize=(20,6))
sns.pointplot(x='Width',y= 'Cost', data = dataset, join=False,palette='rainbow')


# From above pointplot
# 
# 1. Most of the points are around 0.
# 2. Very few points above 0.

# # Count Plot

# 1. A countplot is kind of like a histogram or a bar graph for some categorical area.
# 
# 2. It simply shows the number of occurrences of an item based on a certain type of category.
# 

# In[ ]:


# count plot of whole datset based on Artist Reputation
plt.figure(figsize=(20,6))
sns.countplot(x= 'Artist Reputation', data =train_dataset, palette='rainbow')


# From above count plot
# 
# distribution of values over complete dataset is symmetrical.

# In[ ]:


# count plot of whole datset based on Height
plt.figure(figsize=(20,6))
sns.countplot(x= 'Height', data =train_dataset, palette='rainbow')


# From above count plot
# 
# distribution of values over complete dataset are skewed left.

# In[ ]:


# count plot of whole datset based on Width
plt.figure(figsize=(20,6))
sns.countplot(x= 'Width', data =train_dataset, palette='rainbow')


# From above count plot
# 
# distribution of values over complete dataset are skewed left.

# In[ ]:


# count plot of whole datset based on Weight
plt.figure(figsize=(20,6))
sns.countplot(x= 'Weight', data =train_dataset, palette='rainbow')


# From above count plot
# 
# distribution of values over complete dataset are skewed left., multimodal

# In[ ]:


# count plot of whole datset based on Price Of Sculpture
plt.figure(figsize=(20,6))
sns.countplot(x= 'Price Of Sculpture', data =train_dataset, palette='rainbow')


# From above count plot
# 
# distribution of values over complete dataset are skewed left, multimodal.

# In[ ]:


# count plot of whole datset based on Base Shipping Price

plt.figure(figsize=(20,6))
sns.countplot(x= 'Base Shipping Price', data =train_dataset, palette='rainbow')


# From above count plot
# 
# distribution of values over complete dataset are skewed left, multimodal.

# In[ ]:


# count plot of whole datset based on Cost
plt.figure(figsize=(20,6))
sns.countplot(x= 'Cost', data =train_dataset, palette='rainbow')


# From above count plot
# 
# distribution of values over complete dataset are multi model that is more than one peak.

# # Strip Plot

# A strip plot is a graphical data anlysis technique for summarizing a univariate data set. The strip plot consists of:
# 
#     1. Horizontal axis = the value of the response variable;
#     2. Verticalal axis = all values are set to 1.
# 
# That is, a strip plot is simply a plot of the sorted response values along one axis. The strip plot is an alternative to a histogram or a density plot. It is typically used for small data sets (histograms and density plots are typically preferred for larger data sets). 

# In[ ]:


# strip plot between Artist Reputation and Height 
plt.figure(figsize=(20,6))
sns.stripplot(x='Artist Reputation', y='Height', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Height with repect to Artist Reputation is between 0 to 50.
# 2. few values are above 50

# In[ ]:


# strip plot between Artist Reputation and Width columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Artist Reputation', y='Width', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Width with repect to Artist Reputation is between 0 to 30.
# 2. few values are above 30

# In[ ]:


# strip plot between Artist Reputation and Weight columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Artist Reputation', y='Weight', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Weight with repect to Artist Reputation are between 0 and 0.1.
# 2. few values are above 0.1.

# In[ ]:


# strip plot between Artist Reputation and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Artist Reputation', y='Price Of Sculpture', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Price of Sculpture with repect to Artist Reputation is between 0 to 1000.
# 2. few values are above 1000

# In[ ]:


# strip plot between Artist Reputation and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Artist Reputation', y='Base Shipping Price', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Base shipping price with repect to Artist Reputation is between 0 to 30.
# 2. few values are between 30 to 100

# In[ ]:


# strip plot between Artist Reputation and Cost columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Artist Reputation', y='Cost', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Cost with repect to Artist Reputation are around 0
# 2. few values are above 0

# In[ ]:


# strip plot between Height and Width columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Height', y='Width', data=train_dataset)


# from above strip plot:
# 
# 1. There is growth in the width values as we move right on height axis.

# In[ ]:


# strip plot between Height and Weight columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Height', y='Weight', data= train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Weight with repect to Height are between 0 to 0.1.
# 2. few values are above 0.1

# In[ ]:


# strip plot between Height and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Height', y='Price Of Sculpture', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Price Of Sculpture with repect to Height is between 0 to 1000.
# 2. few values are above 1000.

# In[ ]:


# strip plot between Height and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Height', y='Base Shipping Price', data=train_dataset)


# from above strip plot:
# 
# 1. Maximum of the distribution of Base Shipping Price with repect to Height is between 0 to 30.
# 2. remaining values are between 30 and 100.

# In[ ]:


# strip plot between Height and Cost columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Height', y='Cost', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Cost with repect to Height are between 0 and 0.1.
# 2. very few values are above 0.1.

# In[ ]:


# strip plot between Width and Weight columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Width', y='Weight', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Weight with repect to Width are between 0 to 0.1.
# 2. few values are above 0.1

# In[ ]:


plt.figure(figsize=(20,6))
sns.stripplot(x='Width', y='Price Of Sculpture', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Price Of Sculpture with repect to Width is between 0 to 1000.
# 2. few values are above 1000.

# In[ ]:


# strip plot between Width and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Width', y='Base Shipping Price', data=train_dataset)


# from above strip plot:
# 
# 1. Maximum of the distribution of Base Shipping Price with repect to Width is between 0 to 30.
# 2. remaining values are between 30 and 100.

# In[ ]:


# strip plot between Width and Cost columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Width', y='Cost', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Cost with repect to Width are between 0 and 0.1.
# 2. very few values are above 0.1.

# In[ ]:


# strip plot between Weight and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Weight', y='Price Of Sculpture', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Price Of Sculpture with repect to Weight is between 0 to 1000.
# 2. But as the weight increases the Price Of Sculpture increases above 1000.

# In[ ]:


# strip plot between Weight and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Weight', y='Base Shipping Price', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Base shipping Price with repect to Weight is between 0 to 30 for 50% weigth values.
# 2. But as the weight increases abve 50% the Base shipping Price increases and range bwtween 30 to 100.

# In[ ]:


# strip plot between Weight and Cost columns
plt.figure(figsize=(20,6))
sns.stripplot(x='Weight', y='Cost', data=train_dataset)


# from above strip plot:
# 
# 1. Most of the distribution of Price Of Sculpture with repect to Weight is between 0 to 1000.
# 2. But as the weight increases the Price Of Sculpture increases above 1000.

# # Swarm Plot
# 
# 
# 
# 1. The swarm plot is a type of scatter plot, but helps in visualizing different categorical variables.
# 
# 2. Scatter plots generally plots based on numeric values, but most of the data analyses happens on categorical variables. So, swarm plots seem very useful in those cases.
# 

# In[ ]:


# swarm plot for Artist Reputation and Height columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Artist Reputation', y= 'Height', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Height with repect to Artist Reputation is between 0 to 50.
# 2. few values are above 50

# In[ ]:


# swarm plot for Artist Reputation and Width columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Artist Reputation', y= 'Width', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Width with repect to Artist Reputation is between 0 to 30.
# 2. few values are above 30

# In[ ]:


# swarm plot for Artist Reputation and Weight columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Artist Reputation', y= 'Weight', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Weight with repect to Artist Reputation are between 0 and 0.1.
# 2. few values are above 0.1.

# In[ ]:


# swarm plot for Artist Reputation and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Artist Reputation', y= 'Price Of Sculpture', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Price of Sculpture with repect to Artist Reputation is between 0 to 1000.
# 2. few values are above 1000

# In[ ]:


# swarm plot for Artist Reputation and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Artist Reputation', y= 'Base Shipping Price', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Base shipping price with repect to Artist Reputation is between 0 to 30.
# 2. few values are between 30 to 100

# In[ ]:


# swarm plot for Artist Reputation and Cost columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Artist Reputation', y= 'Cost', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Cost with repect to Artist Reputation are around 0
# 2. few values area bove 0

# In[ ]:


# swarm plot for Height and Width columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Height', y='Width', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. There is growth in the width values as we move right on height axis.

# In[ ]:


# swarm plot for Height and Weight columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Height', y='Weight', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Weight with repect to Height are between 0 to 0.1.
# 2. few values are above 0.1

# In[ ]:


# swarm plot for Height and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Height', y='Price Of Sculpture', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Price Of Sculpture with repect to Height is between 0 to 1000.
# 2. few values are above 1000.

# In[ ]:


# swarm plot for Height and Base Shipping Price columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Height', y='Base Shipping Price', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Maximum of the distribution of Base Shipping Price with repect to Height is between 0 to 30.
# 2. remaining values are between 30 and 100.

# In[ ]:


# swarm plot for Height and Cost columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Height', y='Cost', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Cost with repect to Height are between 0 and 0.1.
# 2. very few values are above 0.1.

# In[ ]:


# swarm plot for Weight and Price Of Sculpture columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Weight', y='Price Of Sculpture', data = train_dataset, palette='rainbow')


# from above swarm plot:
# 
# 1. Most of the distribution of Price Of Sculpture with repect to Weight is between 0 to 1000.
# 2. But as the weight increases the Price Of Sculpture increases above 1000.

# In[ ]:


# swarm plot for Weight and Cost columns
plt.figure(figsize=(20,6))
sns.swarmplot(x= 'Weight', y='Cost', data = train_dataset, palette='rainbow')


# 1. Most of the distribution of Cost with repect to Weight is between 0 to 0.1.
# 2. But as the weight increases the Price Of Sculpture increases above 0.1

# # Combine Plots
# combination of boxen plot and swarm plot.
# Just to see the distribution of both graphs in one.

# In[ ]:


# combine boxen and swarm plot for Artist Reputation and Height columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Artist Reputation', y= 'Height', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Artist Reputation', y= 'Height', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Most of the distribution of Height with repect to Artist Reputation is between 0 to 50.
# 2. few values are above 50

# In[ ]:


# combine boxen and swarm plot for Artist Reputation and Width columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Artist Reputation', y= 'Width', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Artist Reputation', y= 'Width', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Most of the distribution of Width with repect to Artist Reputation is between 0 to 30.
# 2. few values are above 30
# 
# 

# In[ ]:


# combine boxen and swarm plot for Artist Reputation and Weight columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Artist Reputation', y= 'Weight', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Artist Reputation', y= 'Weight', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above Combine plot:
# 
# 1. Most of the distribution of Weight with repect to Artist Reputation are between 0 and 0.1.
# 2. few values are above 0.1.

# In[ ]:


# combine boxen and swarm plot for Artist Reputation and Price Of Sculpture columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Artist Reputation', y= 'Price Of Sculpture', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Artist Reputation', y= 'Price Of Sculpture', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Most of the distribution of Price of Sculpture with repect to Artist Reputation is between 0 to 1000.
# 2. few values are above 1000

# In[ ]:


# combine boxen and swarm plot for Artist Reputation and Base Shipping Price columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Artist Reputation', y= 'Base Shipping Price', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Artist Reputation', y= 'Base Shipping Price', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Most of the distribution of Base shipping price with repect to Artist Reputation is between 0 to 30.
# 2. few values are between 30 to 100

# In[ ]:


# combine boxen and swarm plot for Artist Reputation and Cost columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Artist Reputation', y= 'Cost', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Artist Reputation', y= 'Cost', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Most of the distribution of Cost with repect to Artist Reputation are around 0
# 2. few values area bove 0

# In[ ]:


# combine boxen and swarm plot for Height and Width columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Height', y= 'Width', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Height', y= 'Width', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. There is growth in the width values as we move right on height axis.
# 

# In[ ]:


# combine boxen and swarm plot for Height and Weight columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Height', y= 'Weight', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Height', y= 'Weight', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Most of the distribution of Weight with repect to Height are between 0 to 0.1.
# 2. few values are above 0.1
# 

# In[ ]:


# combine boxen and swarm plot for Height and Price Of Sculpture columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Height', y= 'Price Of Sculpture', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Height', y= 'Price Of Sculpture', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Most of the distribution of Price Of Sculpture with repect to Height is between 0 to 1000.
# 2. few values are above 1000.
# 
# 
# 

# In[ ]:


# combine boxen and swarm plot for Height and Base Shipping Price columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Height', y= 'Base Shipping Price', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Height', y= 'Base Shipping Price', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Maximum of the distribution of Base Shipping Price with repect to Height is between 0 to 30.
# 2. remaining values are between 30 and 100.

# In[ ]:


# combine boxen and swarm plot for Height and Cost columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Height', y= 'Cost', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Height', y= 'Cost', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. Most of the distribution of Cost with repect to Height are between 0 and 0.1.
# 2. very few values are above 0.1.
# 

# In[ ]:


# combine boxen and swarm plot for Width and Weight columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Width', y= 'Weight', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Width', y= 'Weight', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. There is very weak relation between Width and Weight

# In[ ]:


# combine boxen and swarm plot for Width and Base Shipping Price columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Width', y= 'Base Shipping Price', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Width', y= 'Base Shipping Price', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. There is weak correlation between Width and Base Shipping Price

# In[ ]:


# combine boxen and swarm plot for Width and Cost columns
plt.figure(figsize=(20,6))

ax = sns.swarmplot(x= 'Width', y= 'Cost', data = train_dataset, palette='rainbow')
ax = sns.boxenplot(x= 'Width', y= 'Cost', data = train_dataset, palette='rainbow')
plt.show()
fig = ax.get_figure()
plt.figure()


# from above combine plot:
# 
# 1. There is weak correlation between Width and cost Price also there are many outliers

# # Dendrogram

# The dendrogram is a visual representation of the compound correlation data. The individual compounds are arranged along the bottom of the dendrogram and referred to as leaf nodes. Compound clusters are formed by joining individual compounds or existing compound clusters with the join point referred to as a node.

# In[ ]:


# Plot a Dendrogram on the columns of the dataset

X = train_dataset.dropna()
import scipy
from scipy.cluster import hierarchy as hc


corr = np.round(scipy.stats.spearmanr(X).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels= X.columns, orientation='left', leaf_font_size=16)
plt.show()


# observation from dendrogram:
# 
# Price Of Scuplture , Weight, Base Shipping Price, Cost , Width Height are all correlated.
# 
# Cost depends on Price Of Scuplture , Weight, Base Shipping Price.

# ### What we learn after plotting the dataset
# 1. Data is very skewed in nature except for `Artist Reputation`  
# 1. There are also outliers in most of the columns  
# 1. We need to make the distribution of all the columns normal in nature 

# # Scaling

# In[ ]:


# Calculating the mean and median of the columns Artist Reputation, Height, Width, Weight
Artist_Reputation_mean = train_dataset['Artist Reputation'].mean()
Height_mean = train_dataset['Height'].mean()
Width_mean = train_dataset['Width'].mean()
Weight_median = train_dataset['Weight'].median()


# In[ ]:


# Making a copy of the dataset 
train_data_copy = train_dataset.copy()

# Extracting State code from the Customer Location column E.g. New Michelle, OH 50777 - State Code: OH 
train_data_copy['State Code'] = train_data_copy['Customer Location'].str[-8:-6]


# ### Why missing values has to handled properly?
# 
# * Having missing values in your data is not necessarily a setback.  
# * Still, it is an opportunity to perform the right feature engineering to guide the model to interpret the missing information the right way.  
# * There are machine learning algorithms and packages that can automatically detect and deal with missing data.  
# * However, it’s still recommended to transform the missing data manually through analysis and coding strategy.   
# * Even in a well-designed and controlled study, missing data occurs in almost all research. Missing data can reduce the statistical power of a study and can produce biased estimates, leading to invalid conclusions.
# 

# ### Why scaling is necessary?

# * Most of the times, your dataset will contain features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Euclidean distance between two data points in their computations, this is a problem.
# * If left alone, these algorithms only take in the magnitude of features neglecting the units. 
# * The results would vary greatly between different units, 5kg and 5000gms. 
# * The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes. 
# * To suppress this effect, we need to bring all features to the same level of magnitudes. This can be achieved by scaling.

# #### Scaling  Feature With Log Transform

# In all of teh below cells we used np.log for scaling the features.
# It is because the log transformation reduces or removes the skewness of our original data. It deals with large range of number 
# for example:
# 
# log(10) = 1
# 
# log(100) = 2
# 
# log(10000) = 4 and so on.
# 
# It reduces these larger ranged values in dataset to small rnge numbers making it normal distribution/less-skewed distribution

# In[ ]:


train_datas4 = train_dataset.copy()


# In[ ]:


# preparing data before feeding the model
# Helper function for preparing data with parameter datframe.
def prepare_data(df):
  import re
  # Drop the irrelevant columns from the dataset
  df.drop(['Customer Id','Artist Name'], axis=1, inplace= True)

  # Filling missing values in the 'Artist Reputation' column
  df['Artist Reputation_NA'] = np.where(df['Artist Reputation'].isnull(),1,0)
  df['Artist Reputation'].fillna(Artist_Reputation_mean, inplace=True)
  
  
  # Filling missing values in the 'Transport' column
  df['Transport_NA'] = np.where(df['Transport'].isnull(),1,0)
  df['Transport'].fillna(df['Transport'].mode()[0], inplace=True)
 
 
  

  # Filling missing values in the 'Remote Location' column
  df['Remote_Location_NA'] = np.where(df['Remote Location'].isnull(),1,0)
  df['Remote Location'].fillna(df['Remote Location'].mode()[0], inplace=True)
  

  # Filling missing values in the 'Height' column
  df['Height_NA'] = np.where(df['Height'].isnull(),1,0)
  df['Height'].fillna(Height_mean, inplace=True)
  
  # Scaling the value in the 'Height' column
  df['Height'] = np.exp(df['Height'])
  
  
  # Filling missing values in the 'Width' column
  df['Width_NA'] = np.where(df['Width'].isnull(),1,0)
  df['Width'].fillna(Width_mean, inplace= True)
  
  
  # Scaling the value in the 'Width' column
  df['Width'] = np.log(df['Width'])
  
  
  # Filling missing values in the 'Weight' column
  df['Weight_NA'] = np.where(df['Weight'].isnull(),1,0)
  df['Weight'].fillna(Weight_median, inplace= True)
  
  
  # Scaling the value in the 'Weight' column
  df['Weight'] = np.log(df['Weight'])
  
  # Filling missing values in the 'Material' column
  df['Material_NA'] = np.where(df['Material'].isnull(),1,0)
  df['Material'].fillna('NA', inplace= True)
  
  
  # Scale the value in the 'Price Of Sculpture' column
  df['Price Of Sculpture'] = np.log(df['Price Of Sculpture'])

  # Using column 'Delivery Date' and 'Scheduled Date' new column is made which the number of days it took to ship the sculpture i.e both column values difference
  df['Delivery Date'] = pd.to_datetime(df['Scheduled Date'], format='%m/%d/%y')
  df['Scheduled Date'] = pd.to_datetime(df['Delivery Date'], format='%m/%d/%y')
  df['del_date_sch_date_diff'] = (df['Scheduled Date'] - df['Delivery Date']).dt.days
  df['del_date_sch_date_diff'] = np.abs(df['del_date_sch_date_diff'])

  # Drop the 'Delivery Date' and 'Scheduled Date' columns after the new column is made
  df.drop(['Delivery Date','Scheduled Date'],axis=1, inplace=True)
  
  
  # Extracting city, state code and pin from the 'Customer Location' column and making new columns for each
  df['city'] = df['Customer Location'].str.split(',',  expand=True)[0]
  df['State_Code'] = df['Customer Location'].str.split(',',  expand=True)[1].str.slice(0,3)
  df['pin'] = df['Customer Location'].str.split(',',  expand=True)[1].str.split(' ',  expand=True)[2]

  city_others = df[df['State_Code'].isna()]['Customer Location'].str.split(' ',  expand=True)[0]
  city_others.index = df[df['State_Code'].isnull()].index
  
  State_code_others = df[df['State_Code'].isna()]['Customer Location'].str.split(' ',  expand=True)[1]
  State_code_others.index = df[df['State_Code'].isnull()].index
  
  pin_others = df[df['pin'].isna()]['Customer Location'].str.split(' ',  expand=True)[2]
  pin_others.index = df[df['pin'].isnull()].index
  

  df.loc[df['State_Code'].isnull(), 'city'] = city_others
  df.loc[df['State_Code'].isnull(), 'State_Code'] = State_code_others
  df.loc[df['pin'].isnull(), 'pin'] = pin_others
 

  # Dropping the columns after their need is over
  df.drop(['Customer Location'], axis= 1, inplace=True)
  df.drop(['city','pin'],  axis= 1, inplace=True)
 
 
  # return the prepared dataframe  
  return df


# In[ ]:


# passing the train data to prepare data function.
train_data = prepare_data(train_datas4)


# ### Now it's much more better than before data preparation
# 

# In[ ]:


# Split the labels and the target
X = train_data.drop(['Cost'],axis=1)
y = np.log(np.abs(train_data['Cost']))


# ### For doing some data transformation like Ordinal Encoding and One Hot Encoding we will use one library called *feature_engine*.  
# Documentation: [Feature Engine](https://feature-engine.readthedocs.io/en/1.1.x/)

# In[ ]:


# Install feature_engine library
get_ipython().system('pip install feature_engine')


# ### One Hot Encoding Explanation

# For categorical variables where no ordinal relationship exists, the integer encoding may not be enough, at best, or misleading to the model at worst.
# 
# Forcing an ordinal relationship via an ordinal encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).
# 
# In this case, a one-hot encoding can be applied to the ordinal representation. This is where the integer encoded variable is removed and one new binary variable is added for each unique integer value in the variable.

# In[ ]:


# Perform One-Hot Encoding

# Import OneHotEncoder from the feature_engine library 
from feature_engine.encoding import OneHotEncoder

# Instantiate the OneHotEncoder
encoder = OneHotEncoder(top_categories=None)

# Fit the encoder with data
encoder.fit(X)

# Transform the data according to the fit
train_X = encoder.transform(X)


# ### Splitting the data into train and test set

# In[ ]:


# Import train_test_split from sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X,y, test_size = 0.2,  train_size=0.8)


# In[ ]:


X_train.head()


# ### Building the machine learning model on the processed dataset

# In[ ]:


# importing necessary libraries for geting metrics of models
import math
import sklearn.metrics as metrics
from sklearn.metrics import median_absolute_error

# Function for calculating RMSE 
def rmse(x,y): 
    return math.sqrt(((x-y)**2).mean())

# Function for calculating all the relevant metrics 
def print_score(m):
    res = [rmse(m.predict(X_train), y_train),rmse(m.predict(X_test), y_test),
           m.score(X_train, y_train),m.score(X_test, y_test),
           median_absolute_error(m.predict(X_train), y_train),median_absolute_error(m.predict(X_test), y_test),
           metrics.mean_absolute_error(m.predict(X_train), y_train),metrics.mean_absolute_error(m.predict(X_test), y_test),
          
          ]
    

    print("RMSE-Train: " + str(res[0]) + "\nRMSE-Test: " + str(res[1]) + "\nScore-Train: " + str(res[2]) + "\nScore-Test: " + str(res[3]) +
         "\nMedAE-Train: " + str(res[4]) + "\nMedAE-Test: " + str(res[5]) + "\nMeanAE-Train: " + str(res[6]) + "\nMeanAE-Test: " + str(res[7]))


# In[ ]:


# Visualize importance of all the features in the dataset for the prediction

def visualize_importance(feature_importances, feat_train_df):
    feature_importance_df = pd.DataFrame()
     # creating dataframe for feature name and feature importance
    _df = pd.DataFrame()
    _df['feature_importance'] = feature_importances
    _df['column'] = feat_train_df.columns
    feature_importance_df = pd.concat([feature_importance_df, _df], 
                                      axis=0, ignore_index=True)
    # grouping all data and sorting in descending order
    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]
    # ploting feature importance data using boxenplot
    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x='feature_importance', 
                  y='column', 
                  order=order, 
                  ax=ax, 
                  palette='viridis', 
                  orient='h')
    ax.tick_params(axis='x', rotation=0)
    ax.set_title('Importance')
    ax.grid()
    fig.tight_layout()
    # return fig, ax
    return fig, ax


# ### Linear Regression

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit a Linear Regression model to the train dataset\n\n# Import LinearRegressor\nfrom sklearn.linear_model import LinearRegression\n\n# Instantiate the model\nlr = LinearRegression()\n# Fit the model to the data\nlr.fit(X_train, y_train)\n\n# print score of the model by calling function\nprint_score(lr)\n\n# visualizing the importance of features.\nfig, ax= visualize_importance(lr.coef_,train_X)\n\n')


# ### Random Forest Regressor

# Random forest is a flexible, easy to use machine learning algorithm that produces, even without hyper-parameter tuning, a great result most of the time. It is also one of the most used algorithms, because of its simplicity and diversity.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit a Random Forest Regressor model to the train dataset\n\n# Import RandomForrestRegressor\nfrom sklearn.ensemble import RandomForestRegressor\n\n# Instantiate the model\nrfe = RandomForestRegressor()\n\n# Fit the model to the data\nrfe.fit(X_train,y_train)\n\n# print score of the model by calling function\nprint_score(rfe)\n\n# visualizing the importance of features.\n\nfig, ax= visualize_importance(rfe.feature_importances_,train_X)\n\n\n\n')


# ### KNeighbors Regressor

# KNN regression is a non-parametric method that, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood. The size of the neighbourhood needs to be set by the analyst or can be chosen using cross-validation to select the size that minimises the mean-squared error.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit a K-Neighbour Regressor model to the train dataset\n\nfrom sklearn.neighbors import KNeighborsRegressor\n\n# Instantiate the model\nkrn = KNeighborsRegressor()\n\n# Fit the model to the data\nkrn.fit(X_train,y_train)\n# print score of the model by calling function\nprint_score(krn)\n')


# ### Gradient Boosting Regressor

# Gradient Boosting Algorithm is generally used when we want to decrease the Bias error.
# it builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit a Gradient Boosting Regressor model to the train dataset\n\nfrom sklearn.ensemble import GradientBoostingRegressor\n\n# Instantiate the model\ngbr = GradientBoostingRegressor()\n# Fit the model to the data\ngbr.fit(X_train,y_train)\n# print score of the model by calling function\nprint_score(gbr)\n# visualizing the importance of features.\nfig, ax= visualize_importance(gbr.feature_importances_,train_X)\n')


# ### DecisionTree Regressor

# Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit a Decision Tree Regressor model to the train dataset\n\nfrom sklearn.tree import DecisionTreeRegressor\n\n# Instantiate the model\ndt = DecisionTreeRegressor()\n# Fit the model to the data\ndt.fit(X_train,y_train)\n# print score of the model by calling function\nprint_score(dt)\n# visualizing the importance of features.\nfig, ax= visualize_importance(dt.feature_importances_,train_X)\n')


# ### AdaBoost Regressor

# An AdaBoost regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit a AdaBoost Regressor model to the train dataset\n\n# Import AdaBoostRegressor\nfrom sklearn.ensemble import AdaBoostRegressor\n# Instantiate the model\nabr = AdaBoostRegressor()\n# Fit the model to the data\nabr.fit(X_train, y_train)\n# print score of the model by calling function\nprint_score(abr)\n# visualizing the importance of features.\nfig, ax= visualize_importance(abr.feature_importances_,train_X)\n')


# ### XGBoost

# XGBoost is an ensemble learning method. Sometimes, it may not be sufficient to rely upon the results of just one machine learning model. Ensemble learning offers a systematic solution to combine the predictive power of multiple learners. The resultant is a single model which gives the aggregated output from several models.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit a XGB Regressor model to the train dataset\n\n\nfrom xgboost import XGBRegressor\n# Instantiate the model\nxgbr = XGBRegressor()\n# Fit the model to the data\nxgbr.fit(X_train, y_train)\n# print score of the model by calling function\nprint_score(xgbr)\n# visualizing the importance of features.\nfig, ax= visualize_importance(xgbr.feature_importances_,train_X)\n')


# ### Light Gradient Boosted Machine

# Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fit a lightgbm Regressor model to the train dataset\n\n# Import lightgbm\nimport lightgbm as lgb\n# Instantiate the model\nlgbm = lgb.LGBMRegressor()\n# Fit the model to the data\nlgbm.fit(X_train,y_train)\n# print score of the model by calling function\nprint_score(lgbm)\n# visualizing the importance of features.\nfig, ax= visualize_importance(lgbm.feature_importances_,train_X)\n')


# ### Comparing all the model based on metric

# In[ ]:


# Helper function for calculating rmse using formula

def compare_models(models,names,X_train,y_train,X_test,y_test):
  # the libraries we need
  import sklearn.metrics as metrics
  from sklearn.model_selection import train_test_split


  # now, create a list with the objects 
  data = {'Metric':['rmse','MedAE','MAE','R-squared']}
  df_train = pd.DataFrame(data)
  df_test = pd.DataFrame(data)

  def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())


  for (model,name) in zip(models,names):
    y_pred= model.predict(X_test) # then predict on the test set
    res = [rmse(model.predict(X_train), y_train),rmse(model.predict(X_test), y_test),
              metrics.median_absolute_error(model.predict(X_train), y_train),metrics.median_absolute_error(model.predict(X_test), y_test),
              metrics.mean_absolute_error(model.predict(X_train), y_train),metrics.mean_absolute_error(model.predict(X_test), y_test),
              metrics.r2_score(model.predict(X_train), y_train),metrics.r2_score(model.predict(X_test), y_test)]
    # get metrics of each model, and add to dataframe 
    df_train[name] = [res[0], res[2], res[4], res[6]]
    df_test[name] = [res[1], res[3], res[5], res[7]]
  return df_train,df_test


# In[ ]:


# create a list of objects of models
# list of models object
models = [lr,  rfe, krn, gbr, xgbr, abr]
# list of models name
names =['Linear','Random','knr','Gradient','Xboost','AdaBoost']
# use function for comparing models by passing list of models object, names, train and test data
comp_model_train, comp_model_test = compare_models(models,names,X_train,y_train, X_test, y_test)


# #### RMSE of all model on train and test data

# In[ ]:


# printing rmse comparision of model on train and test

print(comp_model_train[:1])


# In[ ]:


print(comp_model_test[:1])


# #### All metrics on train and test data

# In[ ]:


# printing comparision of model on train  and test set

print(comp_model_train)
print('\n')
print(comp_model_test)


# ## Hyperparameter Tunning

# A hyperparameter is a parameter whose value is set before the learning process begins.
# 
# Hyperparameters tuning is crucial as they control the overall behavior of a machine learning model. 
# 
# Every machine learning models will have different hyperparameters that can be set.

# ### RamdomizedSearchCV

# RandomizedSearchCV is very useful when we have many parameters to try and the training time is very long.
#  1. The first step is to write the parameters that we want to consider
#  2. From these parameters select the best ones.(which are printed in output)

# In[ ]:


# Helper function to perform hyper parameter tunning with RandomizedSearchCV
def random_Search(model,X_train,y_train, param_grid):
  from sklearn.model_selection import RandomizedSearchCV

  # Random search of parameters, using 3 fold cross validation, 
  # search across 100 different combinations, and use all available cores
  random = RandomizedSearchCV(estimator= model, param_distributions=param_grid, n_iter= 10, cv= 3, verbose=2, random_state=42, n_jobs = -1)

  # print best parameters
  random.fit(X_train,y_train)
  random.best_params_


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# create parameters dict in list for tunning\nrf_para_grid ={'n_estimators':[int(x) for x in np.linspace(start=200, stop=400, num = 50)],\n               'max_features':['auto','sqrt'],\n               'max_depth':[int(x) for x in np.linspace(3,9,num=3)],\n               'min_samples_split':[2,5,10],\n               'min_samples_leaf':[1,2,4],\n               'bootstrap':[True,False]\n               }\n# passing data for hyper parameter tunning with RandomSearchCV\nrandom_Search(RandomForestRegressor(),X_train,y_train, param_grid = rf_para_grid)\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# create GradientBoostRegressor parameters dict for tunning\n\nGBR_para_grid = {'n_estimators': [x for x in range(200,500, 50)],\n                 'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5],\n                 'max_depth':[x for x in range(5,17, 3)],\n                 'min_samples_split':[x for x in range(2,10)]\n                 }\n# passing data for hyper parameter tunning with RandomSearchCV\nrandom_Search(GradientBoostingRegressor(),X_train, y_train, param_grid = GBR_para_grid)\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# create parameters dict in list for tunning\nknn_para_grid={'leaf_size': list(range(1,10)),\n               'n_neighbors':list(range(1,20)),\n               'p':[1,2]\n               }\n\n# passing data for hyper parameter tunning with RandomSearchCV\nrandom_Search(KNeighborsRegressor(),X_train,y_train, param_grid = knn_para_grid)\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# create DecisionTreeRegressor parameters dict for tunning\nDTR_para_grid= {'max_depth':[1,3,5,7,9,11,12],\n                'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10],\n                'min_weight_fraction_leaf':[0.1,0.2,0.3,0.4,0.5],\n                'max_features':['auto','log2','sqrt',None],\n                'max_leaf_nodes':[None,10,20,30,40,50,60,70,80,90],\n                'splitter':['best','random']\n                }\n# passing data for hyper parameter tunning with Randomized search cv\n\nfrom sklearn.model_selection import RandomizedSearchCV\n\n# Random search of parameters, using 3 fold cross validation, \n# search across 100 different combinations, and use all available cores\nrandom = RandomizedSearchCV(estimator= DecisionTreeRegressor(), param_distributions=DTR_para_grid, n_iter= 10, cv= 3, verbose=2, random_state=42, n_jobs = -1,error_score='raise')\nrandom.fit(X_train,y_train)\nprint(random.best_params_)\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# create AdaBoostRegressor parameters dict for tunning\n    \n\nAda_para_grid= {\n    'n_estimators':[10,50,100,500],\n    'learning_rate':[0.0001,0.001,0.01,0.1,1.0],\n\n}\n# passing data for hyper parameter tunning with RandomSearchCV\nrandom_Search(AdaBoostRegressor(),X_train,y_train,param_grid = Ada_para_grid)\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# create XGBoostRegressor parameters dict for tunning\n\nXGB_para_grid = {'learning_rate':[ 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],\n                 'max_depth':[ 3, 4, 5, 6, 8, 10, 12, 15],\n                 'min_child_weight':[ 1, 3, 5, 7],\n                 'gamma':[ 0.0, 0.1, 0.2, 0.3, 0.4],\n                 'colsample_bytree':[ 0.3, 0.4, 0.5, 0.7]}\n# passing data for hyper parameter tunning with RandomSearchCV\nrandom_Search(XGBRegressor(),X_train,y_train,param_grid = Ada_para_grid)\n")


# ### NOTE:
# 
# you can use any one of RandomizedSearchCv or GridSearchCV, both works fine.
# 

# ### GridSearchCV

# One traditional and popular way to perform hyperparameter tuning is by using an Exhaustive Grid Search from Scikit learn. 
# 
# This method tries every possible combination of each set of hyper-parameters. 
# 
# Using this method, we can find the best set of values in the parameter search space. 
# 
# This usually uses more computational power and takes a long time to run since this method needs to try every combination in the grid size.a model for each combination of algorithm parameters specified in a grid.

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import GridSearchCV\n\n# create RandomForest parameters dict in list for tunning\n\nrf_para_grid = {'n_estimators':[100,300,500],\n               'max_features':['auto','sqrt'],\n               'max_depth':[int(x) for x in np.linspace(3,9,num=3)],\n               'min_samples_split':[2,5,10],\n               'min_samples_leaf':[1,2,4],\n               'bootstrap':[True,False]\n               }\n\n# Grid search of parameters, using 5 fold cross validation\nmodel_cv = GridSearchCV(estimator= RandomForestRegressor(), param_grid = rf_para_grid,\n                            cv=3,\n                            return_train_score=True, n_jobs = -1,\n                            verbose=2)\n\n#fit model_cv\nmodel_cv.fit(X_train, y_train)\n# print best parameters\nprint(model_cv.best_params_)\n\n# print best score\nprint(model_cv.best_score_)\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Fit a Random Forest Regressor model to the train dataset\n\n# Import RandomForrestRegressor\nfrom sklearn.ensemble import RandomForestRegressor\n\n# Instantiate the model\nrfe = RandomForestRegressor(max_depth= 9, max_features='auto', min_samples_leaf= 4, min_samples_split= 5, n_estimators= 500,bootstrap= True)\n\n# Fit the model to the data\nrfe.fit(X_train,y_train)\n\n# print score of the model by calling function\nprint_score(rfe)\n\n# visualizing the importance of features.\n\nfig, ax= visualize_importance(rfe.feature_importances_,train_X)\n")


# Note: above cell only tune RandomForest, you can create cells and tune  all models using GridSearchCv, parameters are same as of RandomisedSearchCv

# ### Now working with the test dataset provided

# In[ ]:


# create sample submission datframe with Customer Id column from test data
sample_submission = test_dataset[['Customer Id']]


# In[ ]:


# Prepare the test dataset i.e do all the transformation that was done on train dataset by callingthe helper function
test_data = prepare_data(test_dataset)


# In[ ]:


# Perform One Hot Encoding on test dataset
test_data = encoder.transform(test_data)


# In[ ]:


# Check if the shape of the train dataset and test dataset matches
print(train_X.shape)
print(test_data.shape)


# In[ ]:


# Perforn the prediction on the test dataset
X_test = test_data
y_predicted = gbr.predict(X_test)


# In[ ]:


# Create a dataframe with predicted result as data
predicted = pd.DataFrame (data= y_predicted, columns=['Pred_Cost'])
predicted


# In[ ]:


# Scale the target variable
sample_submission['Cost'] =np.exp(y_predicted)


# In[ ]:


# Save the dataframe in the CSV format
sample_submission.to_csv('sample_submission.csv', index=False)


# ### Conclusion
# 
# We did training and prediction using all the above models and selected random forest as final model as it performed well compard to other models with acurracy of 99% on train data and around 96% on test data.
# 
# According to this model, the predicted value we got,
# matches with the actual target values. Does the model is performing well.
# 
# We have performed EDA, preprocessing, build different models, visualized feature importance, did hyper parameter tunning of each model and did prediction.
