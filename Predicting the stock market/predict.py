import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

## read in the data
sphist = pd.read_csv('sphist.csv')
# convert 'date' column to a pandas date type
sphist['Date'] = pd.to_datetime(sphist['Date'])
# sort the dataframe on the 'date' column
sphist = sphist.sort_values('Date', ascending = True)

## Generate indicators
# it's important not to include the current price in indicators!
# we can use a for loop along with the iterrows method to loop over the rows and compute the indicators. Or use some time series tools of pandas, including the rolling function and the shift method.
# the average price from the past 5 days.
day_5 = sphist['Close'].rolling(window = 5,min_periods = 1).mean().shift(periods = 1)
# the average price from the past 30 days
day_30 = sphist['Close'].rolling(window = 30,min_periods = 1).mean().shift(periods = 1)
# the average price from the past 365 days
day_365 = sphist['Close'].rolling(window = 365, min_periods = 1).mean().shift(periods = 1)
# the standard deviation of the price over the past 5 days
std_5 = sphist['Close'].rolling(window = 5,min_periods = 1).std().shift(periods = 1)
# the standard deviation of the price over the past 30 days
std_30 = sphist['Close'].rolling(window = 30,min_periods = 1).std().shift(periods = 1)
# the standard deviation of the price over the past 365 days
std_365 = sphist['Close'].rolling(window = 365,min_periods = 1).std().shift(periods = 1)
# add data to dataframe
sphist['day_5'] = day_5
sphist['day_30'] = day_30
sphist['day_365'] = day_365
sphist['ratio_price_5_365'] = day_5/day_365
sphist['std_5'] = std_5
sphist['std_30'] = std_30
sphist['std_365'] = std_365
sphist['ratio_std_5_365'] = std_5/std_365

## Splitting up the data
# remove any rows from the dataframe that fall before 1951-01-03
print(sphist.shape)
sphist = sphist.drop(index = sphist[sphist['Date'] < datetime(year = 1951, month = 1, day = 3)].index, axis= 0)
print(sphist.shape)
# Use dropna method to remove any rows with NaN values
sphist = sphist.dropna(axis = 0)
print(sphist.shape)
# generate train and test dataframe for algorithm
train = sphist[sphist['Date'] < datetime(year = 2013, month = 1, day = 1)]
test = sphist[sphist['Date'] >= datetime(year = 2013, month = 1, day = 1)]
print(train.shape)
print(test.shape)

## Making predictions
# we are going to use Mean Absolute Error for this project
# choose features and train a linear model
features = ['day_5','day_30','day_365']
target = 'Close'
Lr = LinearRegression()
Lr.fit(train[features],train[target])
predictions = Lr.predict(test[features])
error = test[target]-predictions
mae = error.abs().mean()
print('features: ')
print(features)
print('the mae is: '+ str(mae))
# choose features and train a linear model
features = ['day_5','day_30','day_365','std_5','std_30','std_365']
target = 'Close'
Lr = LinearRegression()
Lr.fit(train[features],train[target])
predictions = Lr.predict(test[features])
error = test[target]-predictions
mae = error.abs().mean()
print('features: ')
print(features)
print('the mae is: '+ str(mae))
features = ['day_5','day_30','day_365','std_5','std_30','std_365','ratio_price_5_365','ratio_std_5_365']
target = 'Close'
Lr = LinearRegression()
Lr.fit(train[features],train[target])
predictions = Lr.predict(test[features])
error = test[target]-predictions
mae = error.abs().mean()
print('features: ')
print(features)
print('the mae is: '+ str(mae))


## More indicators to try:
#-The average volume over the past five days.
#-The average volume over the past year.
#-The ratio between the average volume for the past five days, and the average volume for the past year.
#-The standard deviation of the average volume over the past five days.
#-The standard deviation of the average volume over the past year.
#-The ratio between the standard deviation of the average volume for the past five days, and the standard deviation of the average volume for the past year.
#-The year component of the date.
#-The ratio between the lowest price in the past year and the current price.
#-The ratio between the highest price in the past year and the current price.
#-The year component of the date.
#-The month component of the date.
#-The day of week.
#-The day component of the date.
#-The number of holidays in the prior month.

## More things to do:
#- Accuracy would improve greatly by making predictions only one day ahead. 
#- You can also improve the algorithm used significantly. Try other techniques, like a random forest, and see if they perform better.
#- You can also incorporate outside data, such as the weather in New York City (where most trading happens) the day before, and the amount of Twitter activity around certain stocks.
#- You can also make the system real-time by writing an automated script to download the latest data when the market closes, and make predictions for the next day.
#- Finally, you can make the system "higher-resolution". You're currently making daily predictions, but you could make hourly, minute-by-minute, or second by second predictions. 

