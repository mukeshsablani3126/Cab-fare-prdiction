#!/usr/bin/env python
# coding: utf-8

# In[143]:


#Load libraries
import os
import pandas as pd
import numpy as np
import collections as defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[5]:


#Set The Current working directory
os.chdir("C:/edwisor")


# In[6]:


os.getcwd()


# In[7]:


# loading Train and Test data


# In[8]:



cab_train = pd.read_csv("train_cab.csv")
test = pd.read_csv("test.csv")


# # Data Preprocessing

# In[9]:


cab_train.head()


# In[10]:


test.head()


# In[11]:


cab_train.shape, test.shape


# # Explaratory Data analysis#############################################################

# In[12]:


############################# Find The Datatypes ##############################


# In[13]:



cab_train.dtypes


# In[14]:


############################# convert catagoric to numeric ####################
cab_train['fare_amount'] = cab_train['fare_amount'].apply(pd.to_numeric, errors='coerce')


# In[15]:


############################# Convert object to datetime ###################### 
from datetime import datetime
import calendar
cab_train.pickup_datetime = pd.to_datetime(cab_train.pickup_datetime, errors='coerce')


# In[16]:



############## get info of all attributes ##################
cab_train.info()


# In[17]:


######################## Split Data time ##########################
cab_train['pickup_year']= cab_train['pickup_datetime'].dt.year
cab_train['pickup_month']=cab_train['pickup_datetime'].dt.month
cab_train['pickup_day_of_month']=cab_train['pickup_datetime'].dt.day
cab_train['pickup_hour']=cab_train['pickup_datetime'].dt.hour
cab_train['pickup_minute']=cab_train['pickup_datetime'].dt.minute
cab_train['pickup_second']=cab_train['pickup_datetime'].dt.second
#cab_train['pickup_weekday_name']=cab_train['pickup_datetime'].dt.weekday_name

cab_train.head()


# In[18]:


###################### Drop pickup_datetime ######################

cab_train = cab_train.drop(['pickup_datetime'], axis=1) 


cab_train = cab_train[cab_train['pickup_longitude']!=0]
cab_train = cab_train[cab_train['pickup_latitude']!=0]
cab_train = cab_train[cab_train['dropoff_longitude']!=0]
cab_train = cab_train[cab_train['dropoff_latitude']!=0]


# In[19]:


###################### Calutlate the distance using haversine formula #######################

from math import sin, cos, sqrt, atan2, radians,asin

#Calculate the great circle distance between two points on the earth (specified in decimal degrees)

def haversine_np(lon1, lat1, lon2, lat2):
    
    # Convert latitude and longitude to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Find the differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Apply the formula 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    
    # Calculate the angle (in radians)
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Convert to kilometers
    km = 6367 * c
    
    return km


# In[20]:


cab_train['Trip_distance_KM'] =  haversine_np(cab_train['pickup_longitude'], cab_train['pickup_latitude'],
                        cab_train['dropoff_longitude'], cab_train['dropoff_latitude']) 


# In[21]:



################## Trip_distance_KM describtion ################

cab_train["Trip_distance_KM"].describe()


# # EDA Analysis-Visualization

# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:



##################### Visualisatin of Trip Fare ###################
plt.figure(figsize=(14,7))
sns.kdeplot(cab_train['fare_amount']).set_title("Visualisation of Trip Fare")
cab_train.loc[cab_train['fare_amount']<0].shape
cab_train["fare_amount"].describe()


# In[24]:


########### Trip fare should not be -ve so here we drop the -ve values ####################
cab_train=cab_train.loc[cab_train['fare_amount']>0]
cab_train.shape


# In[25]:


###################################### Visualisation of passenger_count ################################
plt.figure(figsize=(14,7))
sns.kdeplot(cab_train['passenger_count']).set_title("Visualisation of passenger_count")


# In[26]:



################################ Passenger count is in between 1 to 6 ########################
cab_train=cab_train[cab_train['passenger_count']<=6]
cab_train=cab_train[cab_train['passenger_count']>=1]
cab_train["passenger_count"].describe()


# In[27]:


############################## Bar plot of fare amount vs passenger count ####################################
plt.figure(figsize=(20,5))
sns.barplot(x='passenger_count',y='fare_amount',data=cab_train).set_title(" Fare Amount vs passenger count")


# In[28]:


############################## Bar plot of fare amount vs Pickup year ####################################
plt.figure(figsize=(20,5))
sns.barplot(x='pickup_month',y='fare_amount',data=cab_train).set_title(" Fare Amount vs month")


# In[29]:


## Trip distance vs fare amount
plt.scatter(x=cab_train['Trip_distance_KM'],y=cab_train['fare_amount'])
plt.xlabel("Trip Distance")
plt.ylabel("Fare Amount")
plt.title("Trip Distance vs Fare Amount")


# In[30]:


############################ storing the EDA data in df ###########################
df = cab_train
#cab_train = df


# # MISSING VALUE ANALYSIS

# In[31]:




#Create dataframe with missing percentage
missing_val = pd.DataFrame(cab_train.isnull().sum())
print(missing_val)

#Reset index
missing_val = missing_val.reset_index()
print(missing_val)

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
print(missing_val)

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(cab_train))*100
print(missing_val)

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
print(missing_val)


# In[32]:


missing_val


# In[33]:


# Actual = 40.774138
#Mean = 39.914672005073626
#Median = 40.752603
#Mode = 41.25456
cab_train['pickup_latitude'].loc[7]


# In[34]:


#create missing value
cab_train['pickup_latitude'].loc[7] = np.nan


# In[35]:


################ Here We select median method #####################
cab_train['pickup_year']= cab_train['pickup_year'].fillna(cab_train['pickup_year'].median())
cab_train['pickup_month']= cab_train['pickup_month'].fillna(cab_train['pickup_month'].median())
cab_train['pickup_day_of_month']= cab_train['pickup_day_of_month'].fillna(cab_train['pickup_day_of_month'].median())
cab_train['pickup_hour']= cab_train['pickup_hour'].fillna(cab_train['pickup_hour'].median())
cab_train['pickup_minute']= cab_train['pickup_minute'].fillna(cab_train['pickup_minute'].median())
cab_train['pickup_second']= cab_train['pickup_second'].fillna(cab_train['pickup_second'].median())


# In[36]:


#################  MEAN METHOD #################
#cab_train['pickup_latitude']= cab_train['pickup_latitude'].fillna(cab_train['pickup_latitude'].mean())
################# MODE METHOD #################
#cab_train['pickup_latitude']= cab_train['pickup_latitude'].fillna(cab_train['pickup_latitude'].median())
#cab_train = cab_train(KNN(k = 3).complete(cab_train), columns = cab_train.columns)


# # OUTLIERS ANALYSIS 
# 

# In[37]:


########################### CONTINUOUS VARIABLES WITH TARGETVARIABLE#############################
cnames = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','Trip_distance_KM','fare_amount']


# # Box plot analysis

# In[38]:


##### BOX PLOT FOR fare_amount
plt.figure(figsize=(20,7))
plt.boxplot(cab_train['fare_amount'])
plt.xlabel('fare_amount')
plt.ylabel('Count')
plt.title("BoxPlot of fare_amount")


# In[39]:


##### BOX PLOT FOR TRIP DISTANCE IN KM
plt.figure(figsize=(21,7))
plt.boxplot([cab_train['Trip_distance_KM']])
plt.title('BOXPLOT METHOD')
plt.xlabel(['Trip_distance_KM'])
plt.ylabel('count')


# In[40]:


cnames = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','Trip_distance_KM']


# In[41]:


############################## Box plot method to remove the outliers ##############################
for i in cnames:
         print(i)
         q75, q25 = np.percentile(cab_train.loc[:,i], [75 ,25])
         print("75% ="+ str(q75))
         print("25% ="+ str(q25))   
         iqr = q75 - q25
         print("IQR ="+ str(iqr))
         min = q25 - (iqr*1.5)
         max = q75 + (iqr*1.5)
         print("Min="+ str(min))
         print("Max="+ str(max))


# In[42]:


# To remove the Outliers                    
cab_train = cab_train.drop(cab_train[cab_train.loc[:,i] < min].index)
cab_train = cab_train.drop(cab_train[cab_train.loc[:,i] > max].index)
   


# # FEATURE SELECTION 
# 

# In[43]:


########################### CONTINUOUS VARIABLES WITH TARGET VARIABLE#############################
cnames = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','Trip_distance_KM']


# In[44]:


#Correlation analysis
df_corr = cab_train.loc[:,cnames]
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(13, 12))
#Generate correlation matrix
corr = df_corr.corr()
#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True, square=True, ax=ax)
plt.plot()


# In[45]:


#Here we can easily conclude dropoff_lattitude highly negative correlation with pickup_longitude
#Trip_distance_KM shows positive correlation

#  we can easily say pickup_latitude,pickup_longitude','dropoff_longitude','dropoff_latitude' are highly correalated with each other


# In[46]:


########################## CATAGARICAL VARIABLES ###############################
cat_names = ['pickup_year','pickup_month','pickup_day_of_month','pickup_hour','pickup_minute','pickup_second','passenger_count']
#Here we are using ANOVA test for catagorical attributes  
for i in cat_names:
    f, p = stats.f_oneway(cab_train[i], cab_train["fare_amount"])
    print("P value for variable "+str(i)+" is "+str(p))
    print("f value for variable "+str(i)+" is "+str(f))
    


# In[47]:


# drop the variables those who are highly correlated
cab_train = cab_train.drop(['pickup_latitude','pickup_longitude','dropoff_longitude','dropoff_latitude'], axis=1)
P = cab_train
P.to_csv("train_Visualisation.csv",index=False)


# # FEATURE SCALING

# In[48]:



#Normality check
plt.figure(figsize=(20,7))
plt.hist(cab_train['Trip_distance_KM'], bins='auto')
plt.xlabel('Trip_distance_KM')
plt.ylabel('Count')
plt.title('Histogram to check normality')


# In[49]:


# As our variables are left sckew here we select Normalisation method 
cnames = ['Trip_distance_KM', 'fare_amount']
for i in cnames:
    print(i)
    if i == 'fare_amount':
        continue
    cab_train[i] = (cab_train[i] - cab_train[i].min())/(cab_train[i].max()-cab_train[i].min())


# In[101]:


#save data of preprocessing in pf
cab_train.to_csv("train_sample1.csv",index=False)
pf= cab_train
cab_train.shape
#cab_train = pf


# In[103]:


#Dummy Variables
###################### Get dummy variables for categorical variables  ##################################
df = pd.get_dummies(data = cab_train, columns = cat_names)
df = df.drop(['passenger_count_1.3'], axis=1)
df.shape
#df_A = df


# In[104]:


df_A = df
df.to_csv("train_sample.csv",index=False)


# # MODEL DEVELOPMENT

# In[105]:


from sklearn.model_selection import train_test_split
y = df['fare_amount']
df.drop(['fare_amount'], inplace = True, axis=1)
X = df


# In[55]:


#Split data into train and test data set


# In[106]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)
print(pf.shape, X_train.shape, X_test.shape,y_train.shape,y_test.shape)


# # LINEAR REGRESSION

# In[118]:


# Building LinearRegression Model on training Data
from sklearn.linear_model import LinearRegression# For linear regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
lm = LinearRegression()


# In[119]:


# lets fit data
lm.fit(X_train,y_train)
#lets print the intercept 
print("LM Intercept : ", lm.intercept_)
predictions_LR =lm.predict(X_test)
RMSE_LR = np.sqrt(metrics.mean_squared_error(y_test,predictions_LR))
r2_LR = metrics.r2_score(y_test,predictions_LR)
MAE_LR = metrics.mean_absolute_error(y_test,predictions_LR)
def MAPE(y_true,y_pred):
    mape = np.mean(np.abs((y_true-y_pred)/y_true))
    return mape
MAPE_LR = MAPE(y_test,predictions_LR)
LR_Results = {'RMSE_LR':RMSE_LR,'r2_LR':r2_LR,'MAE_LR':MAE_LR,'MAPE_LR':MAPE_LR}
print(LR_Results)


# # DECISION TREE

# In[139]:


from sklearn.tree import DecisionTreeRegressor# For Decision Tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[140]:


# Building Decision Tree Model on training Data
fit_DT  = DecisionTreeRegressor(max_depth = 4).fit(X_train, y_train)
# Apply model on splitted test data
predictions_DT = fit_DT.predict(X_test)
MAE_DT = metrics.mean_absolute_error(y_test,predictions_DT)
RMSE_DT = np.sqrt(metrics.mean_squared_error(y_test,predictions_DT))
r2_DT = metrics.r2_score(y_test,predictions_DT)
MAE_DT = metrics.mean_absolute_error(y_test,predictions_DT)
MAPE_DT = MAPE(y_test,predictions_DT)
DT_Results = {'RMSE_DT':RMSE_DT,'r2_DT':r2_DT,'MAE_DT':MAE_DT,'MAPE_DT':MAPE_DT}
print(DT_Results)


# In[71]:



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[75]:



# Building Random Forest Model on training data
RFModel = RandomForestRegressor(n_estimators = 200).fit(X_train, y_train)
# Apply model on splitting test data
predictions_RF = RFModel.predict(X_test)
RMSE_RF = np.sqrt(metrics.mean_squared_error(y_test,predictions_RF))
r2_RF = metrics.r2_score(y_test,predictions_RF)
MAE_RF = metrics.mean_absolute_error(y_test,predictions_RF)
MAPE_RF = MAPE(y_test,predictions_RF) 
RF_Results = {'RMSE_RF':RMSE_RF,'r2_RF':r2_RF,'MAE_RF':MAE_RF,'MAPE_RF':MAPE_RF}
print(RF_Results)


# In[76]:


## Error Metrics for All the above 3 models
Error_Metrics = {'RMSE':[RMSE_LR,RMSE_DT,RMSE_RF],
                  'r2':[r2_LR,r2_DT,r2_RF],
                     'MAE':[MAE_LR,MAE_DT,MAE_RF],
                   'MAPE':[MAPE_LR,MAPE_DT,MAPE_RF]}
                 

metrics_result =pd.DataFrame(Error_Metrics,index = ['Linear Regression', 'Decision Tree', 'Random Forest']) 

print(metrics_result)


# In[142]:



# Accuracy of linear  Regression  
#MAPE=5.4399, Accuracy=94.56
# Accuracy of Decision Tree
#MAPE=0.4387, Accuracy=99.5613
# Accuracy of Random Forest
#MAPE=0.2700,Accuracy=99.73


# # So we can freeze model Random forest because give high accuracy and also less value of MAPE  among all model.

# In[78]:


# Now, we will manipulate test data
test.describe()


# In[88]:


test.head()


# # Prediction  on Test Data.csv
# 
# 
# Training the model on X, y (train.csv dataset full) and then predicting on predictions_RF.to_csv(copy of test.csv) We are using Random forest algoritham because it has better results than other

# In[89]:


predictions_RF = pd.DataFrame(predictions_RF) 


# In[90]:


predictions_RF.to_csv('test.csv python.csv',header= True , index= False)


# In[91]:


#saving X_test into directory
X_test.to_csv("xtest_final.csv", index = False , header= True)


# In[92]:


df = pd.read_csv('xtest_final.csv')
df


# In[93]:


#joining two dataframes
final_result = pd.concat([df, predictions_RF], axis=1)


# In[94]:


#renaming column name
final_result.rename(columns={0: 'fare_amount'},inplace = True)


# In[144]:


#saving result in csv format
final_result.to_csv('cab_final_result_.csv',header= True , index= False)


# In[ ]:




