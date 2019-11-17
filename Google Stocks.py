# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:46:41 2018

@author: Jake
"""
import pandas as pd 
from pandas import Series as se
from pandas import DataFrame as df
import seaborn as sns
sns.set_style("white") # Plots
import matplotlib.pyplot as plt # Plots
import math
import quandl as qu
from sklearn import preprocessing, cross_validation, svm

from sklearn.linear_model import LinearRegression as lr

googlePrice = qu.get('WIKI/GOOGL')

googlePrice['Adj. Volatility'] = (googlePrice['Adj. High'] - googlePrice['Adj. Low'])/googlePrice['Adj. Open']

googlePrice['Adj. Shift'] = (googlePrice['Adj. Open'] - googlePrice['Adj. Close'])/googlePrice['Adj. Open']

sns.distplot(googlePrice['Adj. Shift'], kde=True)

googlePrice = googlePrice[['Adj. Volatility','Adj. Shift','Adj. Close','Adj. Volume']]

sns.violinplot(data=googlePrice['Adj. Volatility'])

#Shifting price by 10 days
googlePrice['Adj. Forecast'] = googlePrice['Adj. Close'].shift(-1)

#Filling N/A with dummy values
googlePrice.dropna(inplace = True)

googlePrice.info()
googlePrice.head()
googlePrice.describe()

features = googlePrice.drop(['Adj. Forecast'],1)

forecast = googlePrice['Adj. Forecast']

feat_train, feat_test, fore_train, fore_test = cross_validation.train_test_split(features, forecast, test_size = 0.2)

fit = lr()
fit.fit(feat_train,fore_train)
fit.predict(feat_test)
fit.score(feat_test, fore_test)
fit.coef_

coeff_df = DataFrame(feat_train.columns)
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(fit.coef_)

# preview
coeff_df
