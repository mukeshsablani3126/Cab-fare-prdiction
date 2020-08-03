rm(list = ls())
setwd("C:/edwisor")

#Check working directory
getwd()

install.packages("dplyr")
library("dplyr")
install.packages("corrgram")
library("corrgram")
install.packages("car")
library(car)
install.packages("class")
library(class)
install.packages("DMwR")
library(" DMwR")
require(DMwR)
install.packages("ggplot2")
library(ggplot2)
install.packages("caret")
library(caret)
library(lattice)
library(rpart)
library(randomForest)
library(RRF)
library(inTrees)

p = c("ggplot2",  "caret", "randomForest", "unbalanced","C50", "Information", "rpart",
      'sampling', 'DataCombine', 'inTrees',"scales","gplots")
install.packages(p)
lapply(p, require, character.only = TRUE)


if(!require(psych)){install.packages("psych")}
if(!require(cluster)){install.packages("cluster")}
if(!require(fpc)){install.packages("fpc")}

# loading Train and test data
cab_train = read.csv("train_cab.csv")
test = read.csv("test.csv")
sapply(train,function(x)sum(is.na(x)))



###################  Exploratory data analysis ###########################

#Getting the number of variables and obervation in the datasets
dim(cab_train)
dim(test)
#Structure of data
str(cab_train)
str(test)

#creating muissing value dataframe
missing_val = data.frame(apply(cab_train, 2 , function(x){sum(is.na(x))}))

missing_val$Columns = row.names(missing_val)
# create missing value % column
names(missing_val)[1] = "missing_percentage"

#fill missing value %
missing_val$missing_percentage = (missing_val$missing_percentage/nrow(cab_train))*100

#fill missing value in descending order
missing_val = missing_val[order(-missing_val$missing_percentage),]

#rearrange column name
missing_val = missing_val[, c(2,1)]

#write missing value dataframe in disk
write.csv(missing_val, "missing_perc.csv", row.names = F)

cab_train$fare_amount = as.numeric(as.character(cab_train$fare_amount))
cab_train$passenger_count = as.numeric(as.character(cab_train$passenger_count))
cab_train$pickup_longitude = as.numeric(as.character(cab_train$pickup_longitude))
cab_train$pickup_latitude = as.numeric(as.character(cab_train$pickup_latitude))
cab_train$dropoff_longitude = as.numeric(as.character(cab_train$dropoff_longitude))
cab_train$dropoff_latitude = as.numeric(as.character(cab_train$dropoff_latitude))


sum(is.na(cab_train$passenger_count))/length(cab_train$passenger_count)*100
# passenger_count has 0.342% of missing values

sum(is.na(cab_train$fare_amount))/length(cab_train$fare_amount)*100
# fare_amount has 0.155% of missing values

# Now we'll check the outliers in passenger count:                                        
boxplot(cab_train$passenger_count, outcol= "Red")
table(cab_train$passenger_count)
nrow(cab_train[which(cab_train$passenger_count > 6),])
cab_train = cab_train[-which(cab_train$passenger_count> 6),]
cab_train = cab_train[-which(cab_train$passenger_count== 0),]
cab_train = cab_train[-which(cab_train$passenger_count< 1),]
cab_train = cab_train[-which(cab_train$passenger_count== 1.3),]
cab_train = cab_train[-which(cab_train$pickup_longitude == 0),]
cab_train = cab_train[-which(cab_train$dropoff_latitude == 0),]
cab_train = cab_train[-which(cab_train$dropoff_longitude == 0),]

nrow(cab_train[which(cab_train$dropoff_longitude ==0),])

table(cab_train$passenger_count)
sapply(cab_train,function(x)sum(is.na(x)))
summary(cab_train$passenger_count)

###############Starting filling missing values of passenger_count############

# As passenger_count is a factor variable, we'll fill the missing values by using Mode method(most frequent value)
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
result = getmode(cab_train$passenger_count)
print(result)

cab_train$passenger_count[is.na(cab_train$passenger_count)] = result
table(cab_train$passenger_count)

###########Starting filling missing values of fare amount#######       

summary(cab_train$fare_amount)
boxplot(cab_train$fare_amount, outcol= "Red")

###########below is the method to extract the outliers######
val = cab_train$fare_amount[cab_train$fare_amount %in% boxplot.stats(cab_train$fare_amount)$out] 
length(val)

#we found that there are 1358 outliers in fare_amount column.

#To save other information, we shouldn't remove outliers#####
    cab_train$fare_amount[cab_train$fare_amount %in% val]= NA
    summary(cab_train$fare_amount)
   
#created NA in place of outliers
#now fill these NA by using mean, median and KNN
cab_train$fare_amount[7500] #choosing random value....at 7500 location, value is = 7
cab_train$fare_amount[7500] = NA

#mean =  8.902
#median = 8
#KNN = 7.857
library(class)
summary(cab_train$fare_amount)
install.packages("VIM")
library(VIM)
#knn method


cab_train = knnImputation(cab_train, k=7) 

cab_train$fare_amount[7500]


# As we can see that KNN imputes the nearest value, we'll freeze the KNN imputation
summary(cab_train$fare_amount)

#Now we'll handle longitude and latitude columns

# As we can see that KNN imputes the nearest value, we'll freeze the KNN imputation
summary(cab_train$fare_amount)

#Now we'll handle longitude and latitude columns

summary(cab_train$pickup_longitude)
summary(cab_train$pickup_latitude)
summary(cab_train$dropoff_longitude)
summary(cab_train$dropoff_latitude)


#we know that range of latitude is (-90,+90) and range of longitude is (-180, 180)
#After getting the summary, we found that pickup_latitude column has a value beyond the range, hence we'll remove it.

nrow(cab_train[which(cab_train$pickup_latitude > 90),])
cab_train = cab_train[-which(cab_train$pickup_latitude> 90),]

#Now, we'll calculate the distance travelled by using longitude and latitude
#There is a 'haversine' formula to calculate the great-circle distance between two points - he shortest distance over the earth's surface 
#We have to create a function to calculate the distance

degree_to_radian = function(deg){
  (deg * pi) / 180            
}
haversine_formula = function(long1,lat1,long2,lat2){
  long_1_radian = degree_to_radian(long1)
  lat_1_radian = degree_to_radian(lat1)     
  long_2_radian = degree_to_radian(long2)
  lat_2_radian = degree_to_radian(lat2)
  dif_lat = degree_to_radian(lat2 - lat1)       
  dif_long = degree_to_radian(long2 - long1)    
  a = sin(dif_lat/2) * sin(dif_lat/2) + cos(lat_1_radian) * cos(lat_2_radian) * sin(dif_long/2) * sin(dif_long/2)
  c = 2 * asin(sqrt(a))
  R = 6371e3    #radius of earth 
  R * c / 1000 #1000 is used to convert to kilometers
}

cab_train$distance_travelled_in_km = haversine_formula(cab_train$pickup_longitude, cab_train$pickup_latitude, cab_train$dropoff_longitude, cab_train$dropoff_latitude)
# Now, we'll remove the variable which have distance_travelled less than 25m (taking practical scenario)
nrow(cab_train[which(cab_train$distance_travelled_in_km < 0.025),])
cab_train = cab_train[-which(cab_train$distance_travelled_in_km < 0.025),]
##########outliers analysis############
boxplot(cab_train$distance_travelled_in_km, outcol= "Red")
summary(cab_train$distance_travelled_in_km)   

val2 = cab_train$distance_travelled_in_km[cab_train$distance_travelled_in_km %in% boxplot.stats(cab_train$distance_travelled_in_km)$out] 
length(val2)
nrow(cab_train[which(cab_train$distance_travelled_in_km > 3000),])
######we found that there are 1339 outliers in distance_travelled_in_km column
#########To save other information, we shouldn't remove outliers


cab_train$distance_travelled_in_km[cab_train$distance_travelled_in_km %in% val2]= NA
summary(cab_train$distance_travelled_in_km)
#created NA in place of outliers
#now fill these NA by using mean, median and KNN

train$distance_travelled_in_km[780] #choosing random value....at 780 location, value is = 4.439835
train$distance_travelled_in_km[780] = NA
#mean =  2.4772
#median = 2.0057
#KNN = 3.9620

summary(cab_train$distance_travelled_in_km)
train = knnImputation(cab_train, k=7) 
cab_train$distance_travelled_in_km[780]


# As we can see that KNN imputes the nearest value, we'll freeze the KNN imputation
summary(cab_train$distance_travelled_in_km)
class(cab_train$pickup_datetime)
as.character(cab_train$pickup_datetime)
cab_train$pickup_datetime = as.factor(as.character(cab_train$pickup_datetime))
class(cab_train$pickup_datetime)
library(data.table)

cab_train$Date = as.Date(cab_train$pickup_datetime)
x = strptime(cab_train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
cab_train$Time = as.factor(format(x,"%H"))
cab_train$Year = as.factor(format(x, "%Y"))
cab_train$month = as.factor(format(x, "%m"))
cab_train$day   = as.numeric(format(x,"%d"))

cab_train = rename(cab_train, "pickup_hour" = "Time")
cab_train$pickup_hour = as.numeric(format(x,"%H"))
cab_train$pickup_datetime = NULL
cab_train$pickup_date = NULL
cab_train = rename(cab_train, "pickup_date" = "Date")
sum(is.na(df_train$pickup_hour))
class(cab_train$pickup_hour)
df_train = na.omit(cab_train)


df_train$passenger_count = as.factor(as.character(df_train$passenger_count))

#df_train$pickup_hour = as.factor(as.character(df_train$pickup_hour))

df_train = df_train[-which(df_train$fare_amount<= 0),]

head(df_train)
#df_train$pickup_hour = NULL
head(df_train)
###############visualization fare amount vs hour graph#########################3
ggplot(data = df_train, aes(x = pickup_hour,y = fare_amount))+
  geom_bar(stat = "identity",color ="DarkSlateBlue")+
  labs(title = "Fare Amount Vs. hour", x = "hour", y = "Fare")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="black", size=6, angle=45))

head(df_train)

df_train$pickup_sessions[df_train$pickup_hour >= 0 & df_train$pickup_hour <  7]  = "1"
df_train$pickup_sessions[df_train$pickup_hour >= 7 & df_train$pickup_hour < 18]  = "2"
df_train$pickup_sessions[df_train$pickup_hour >= 18 & df_train$pickup_hour <= 23] = "3"
head(df_train)
#As we can see in the histogram, pickup_hour varies in different sessions of day. So we have divided the days in 3 sessions.
# from 12:00am to 6:00 am = day started defined as "1"
# from 07:00am to 5:00 pm = day mid defined as "2"
# from 06:00pm to 11:00 pm = day end defined as "3"

df_train$pickup_hour = NULL
df_train$pickup_date = NULL
head(df_train)
#############visualization veds fare amount veds month graph################3
ggplot(data = df_train, aes(x = month,y = fare_amount))+
  geom_bar(stat = "identity",color ="DarkSlateBlue")+
  labs(title = "Fare Amount Vs. month", x = "month", y = "Fare")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="black", size=6, angle=45))
#There is not very much difference in Cab_fare in different months

########### visualization fare amount wrt. day graph#################
ggplot(data = df_train, aes(x = day,y = fare_amount))+
  geom_bar(stat = "identity",color ="DarkSlateBlue")+
  labs(title = "fare_amount wrt. day", x = "day", y = "Fare")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="black", size=6, angle=45))
#As we can see that cab_fare is lesser in last 3 days of month end as compared to other days so we'll define 2 different categories of days

df_train$pickup_days[df_train$day >= 1 & df_train$day <  29]  = "2"
df_train$pickup_days[df_train$day >= 29]  = "1"
df_train$day = NULL
# "2" for first 28 days and "1" for last 3 days

head(df_train)
########## fare amount veds year graph############
ggplot(data = df_train, aes(x = Year,y = fare_amount))+
  geom_bar(stat = "identity",color ="DarkSlateBlue")+
  labs(title = "Fare Amount Vs. year", x = "Year", y = "fare")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="black", size=6, angle=45))


head(df_train)
library(magrittr)
table(df_train$pickup_sessions)
table(df_train$pickup_days)
##############converting categorical variable into factor##########
df_train$pickup_days %<>% factor
df_train$pickup_sessions %<>% factor
##########feature selection####################################

###save numeric column in single variable##
numeric_index = sapply(df_train, is.numeric)
numeric_data = df_train[, numeric_index]
cnames = colnames(numeric_data)

##########correlation analysis###
library("corrgram")
  corrgram(df_train[, numeric_index], order = F,
           upper.panel = panel.pie, text.panel = panel.txt, main= "Correlation Plot")

factor_index = sapply(df_train, is.factor)
factor_data = df_train[, factor_index]

##########remove highly dependent variable############
df_train$dropoff_longitude = NULL
df_train$dropoff_latitude = NULL
#########feature scaling#################################################################################

#########plot histogram to check the normality of data###########
par("mar")
par(mar=c(4,4,4,4))
hist(df_train$distance_travelled_in_km)
hist(df_train$pickup_longitude)
hist(df_train$pickup_latitude)
###Above histogramme results shows those variable are not normally distributed then we go for normalization of data step. 

###########Normalization of data#####
c_names = c("pickup_latitude", "pickup_longitude", "distance_travelled_in_km")
for (i in c_names) {
  print(i)
  df_train[,i] = (df_train[,i]- min(df_train[i]))/(max(df_train[,i]-min(df_train[i])))
}


head(df_train)

str(df_train)
names(df_train)
cat_var = c("passenger_count", "Year", "month", "pickup_sessions", "pickup_days")

##############Performing ANOVA test#######
  
  aov(df_train$fare_amount~ df_train$Year)
# for all categorical variables
for(i in cat_var){
  print(i)
  Anova_test_result = summary(aov(formula = fare_amount~df_train[,i],df_train))
  print(Anova_test_result)
}
df_train$pickup_days = NULL
head(df_train)

df = cab_train

########check & remove multicollinearity###########
install.packages("sp")
library(sp)
install.packages("raster")
library(raster)
library(usdm)
library(car)
vif(df[,-1])

numeric_index = sapply(df,is.numeric)  #selecting only numeric
numeric_data = df[,numeric_index]#subset of numeric data

 vifcor(numeric_data[,-1], th = 0.9))


#divide into train & test
set.seed(1234)
train_index = sample(1:nrow(df), 0.9 * nrow(df))
train = df[train_index,]
test = df[-train_index,]


###############################Linear Regression model#############################################3
# fit linear regression model
# we will use the lm() function in the stats package
lm_model = lm(fare_amount ~., data =train)

#predictions
summary(lm_model)

# Lets check the assumptions of ols regression 
#Error should follow normal distribution and no hetroscadacity
# assumptions are checked usig residual plot and normal qq plot 
# Change the panel layout to 2 x 2


par(mfrow = c(2, 2))
plot(lm_model)

# No multicolinearity between Independent variables 

vif(df_train[,-1])
vif(lm_model) 
 

###########predicting for splitted test data****************************

predictions_lm = predict(lm_model, test[,-1])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}


print(postResample(pred = predictions_lm , obs =test$fare_amount))

MAPE(test[,1],predictions_lm)
MAPE

#evaluation
regr.eval(test[,1],predictions_lm, stats = c('mape','rmse'))


#MAPE=18.22
#RMSE=0.224
#Accuracy=81.78


#################################Decision tree###########################################
library(rpart)
library(dplyr)
DT = rpart(fare_amount ~ . , data =train , method = "anova")
predictions_DT = predict(DT, test[,-1])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}


print(postResample(pred = predictions_DT , obs =test$fare_amount))

MAPE(test[,1],predictions_DT)
MAPE
#evaluation
regr.eval(test[,1], predictions_DT, stats = c('mape','rmse'))

# MAPE=21.5
# RMSE=0.249
# Accuracy=78.5


############################ Random forest model#############

RF_model = randomForest(fare_amount ~.,data = train, importance = TRUE, ntree = 200)
RF_Predictions = predict(RF_model, test[,-1])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}
print(postResample(pred = RF_Predictions , obs =test$fare_amount))
MAPE(test[,1],RF_Predictions)
MAPE

#MAPE=18.20
#RMSE=0.221
#Accuracy=81.8

##########manipulating test data for prediction##########
  
  
head(test)
df_test = test
df_test$distance_travelled_in_km = haversine_formula(df_test$pickup_longitude, df_test$pickup_latitude, df_test$dropoff_longitude, df_test$dropoff_latitude)
head(df_test)    

as.character(df_test$pickup_datetime)
df_test$pickup_datetime = as.factor(as.character(df_test$pickup_datetime))
class(df_test$pickup_datetime)
df_test$Date = as.Date(df_test$pickup_datetime)
y = strptime(df_test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df_test$pickup_hour = as.factor(format(y,"%H"))
df_test$Year = as.factor(format(y, "%Y"))
df_test$month = as.factor(format(y, "%m"))
df_test$day   = as.numeric(format(y,"%d"))

df_test$pickup_hour = as.numeric(format(y,"%H"))
df_test$pickup_datetime = NULL
df_test$pickup_date = NULL
df_test = rename(df_test, "pickup_date" = "Date")
sum(is.na(df_test$pickup_hour))
class(train$pickup_hour)



head(df_test)
df_test$pickup_sessions[df_test$pickup_hour >= 0 & df_test$pickup_hour <  7]  = "1"
df_test$pickup_sessions[df_test$pickup_hour >= 7 & df_test$pickup_hour < 18]  = "2"
df_test$pickup_sessions[df_test$pickup_hour >= 18 & df_test$pickup_hour <= 23] = "3"
head(df_test)
df_test$pickup_date = NULL
df_test$dropoff_longitude = NULL
df_test$dropoff_latitude = NULL
df_test$pickup_hour = NULL
df_test$day = NULL
head(df_test)
df_test$passenger_count %<>% factor
df_test$month %<>% factor
df_test$pickup_sessions %<>% factor


col_names = c("pickup_latitude", "pickup_longitude", "distance_travelled_in_km")
for (j in col_names) {
  print(j)
  df_test[,j] = (df_test[,j]- min(df_test[j]))/(max(df_test[,j]-min(df_test[j])))
}

#So we can freeze model Random forest because give high accuracy and also less value of MAPE among all model.¶

#############selecting Random Forest for predict test data##########


#predictions
predictions_test = predict(RF_model, df_test)

#save predictions as dataframe
predictions_test = as.data.frame(df_test)

#importing original test dataset
cab_results = read.csv("test.csv", header = T)

#columnbind target results with test data
cab_results = cbind(predictions_test, cab_results)

#renaming column
names(cab_results)[1] = "fare_amount_predicted"

#saving output in csv format
write.csv(cab_results, "fare amount results r.csv", row.names = F)
