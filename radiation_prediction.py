import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import deque
import math
import random
from matplotlib import style
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import KFold, train_test_split, ShuffleSplit,GridSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from datetime import timezone
import time
import csv
from sklearn.naive_bayes import GaussianNB
import pickle
import json

df = pd.read_csv("6.401852_5.615612_SAM_p50.csv")
df = df.dropna()
print(df.shape)
plt.hist(df['DHI(W/m2)'],bins=50,rwidth=0.8)
plt.show()
#print(df.columns)
#print(df.head(10))

df = df.drop(['Year', 'Month', 'Day','Minute','Albedo', 'Azimuth(degree)', 'cloudopacity(%)', 'Dew Point(deg.c)','Snow Depth(cm)'],axis='columns')
df2 = df.copy()
df3 = df.copy()
#print(df.columns)
#print(df.head(10))

# visualizing values and applying feature engineering
#plt.scatter(df_target['DHI(W/m2)'], df_features['Hour'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Hour')
#plt.show()

#plt.scatter(df['DHI(W/m2)'], df['Temperature(deg.C)'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Temperature(deg.C)')
#plt.show()

df['DHI/temp'] = df['DHI(W/m2)']/df['Temperature(deg.C)']
#plt.hist(df['DHI/temp'],rwidth=0.8,bins=50)
#plt.show()

df = df[df['DHI/temp']<20]

#plt.scatter(df['DHI(W/m2)'], df['Temperature(deg.C)'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Temperature(deg.C)')
#plt.show()

df = df.drop(['DHI/temp'],axis='columns')

#plt.scatter(df['DHI(W/m2)'], df['Relative humidity(%)'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Relative humidity(%)')
#plt.show()

df['DHI/Relative humidity'] = df['DHI(W/m2)']/df['Relative humidity(%)']
#plt.hist(df['DHI/Relative humidity'],rwidth=0.8,bins=50)
#plt.show()

df = df[df['DHI/Relative humidity']<10]
#plt.scatter(df['DHI(W/m2)'], df['Relative humidity(%)'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Relative humidity(%)')
#plt.show()

df = df.drop(['DHI/Relative humidity'],axis='columns')

#plt.scatter(df['DHI(W/m2)'], df['Pressure(mbar)'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Pressure(mbar)')
#plt.show()

df['DHI/pressure'] = df['DHI(W/m2)']/df['Pressure(mbar)']
#plt.hist(df['DHI/pressure'],rwidth=0.8,bins=50)
#plt.show()

df = df[df['DHI/pressure']<0.5]

df = df.drop(['DHI/pressure'],axis='columns')

#plt.scatter(df['DHI(W/m2)'], df['Wind Direction(degree)'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Wind Direction(degree)')
#plt.show()

df['DHI/Wind Direction(degree)'] = df['DHI(W/m2)']/df['Wind Direction(degree)']
#plt.hist(df['DHI/Wind Direction(degree)'],rwidth=0.8,bins=50)
#plt.show()

df = df[df['DHI/Wind Direction(degree)']<3]

df = df.drop(['DHI/Wind Direction(degree)'],axis='columns')
#plt.scatter(df['DHI(W/m2)'], df['Wind Direction(degree)'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Wind Direction(degree)')
#plt.show()

#plt.scatter(df['DHI(W/m2)'], df['Wind Speed(m/s)'], marker='+')
#plt.xlabel('DHI')
#plt.ylabel('Wind Direction(degree)')
#plt.show()

scaler = StandardScaler()

# Removing the target columns
df_target = df.drop(['Hour', 'Temperature(deg.C)','Relative humidity(%)', 'Pressure(mbar)', 'Wind Direction(degree)','Wind Speed(m/s)','Zenith(degree)'],axis='columns')
df_features = df.drop(['DHI(W/m2)', 'DNI', 'EBH', 'GHI','Hour'],axis='columns')
print(df_features.columns)
df_features = pd.DataFrame(scaler.fit_transform(df_features), columns = df_features.columns)
y = df_target['DHI(W/m2)']

X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=101)
print(cross_val_score(LinearRegression(),df_features,y))
print(cross_val_score(SVR(),df_features,y))
print(cross_val_score(RandomForestRegressor(),df_features,y))
clf = RandomForestRegressor(n_estimators=150,criterion='friedman_mse',max_features='sqrt')
clf.fit(X_train,y_train)
#with open("Auxiliary_model_DHI.pickle",'wb') as f:
#       pickle.dump(clf,f)
print(clf.score(X_test,y_test)) # 0.8977735334422938
#lpredictions = clf.predict(X_test)
#print(lpredictions)
#print(y_test)

df_to_be_predicted = pd.read_csv('seen.csv')

df_to_be_predicted = df_to_be_predicted.drop(['total_precipitation_sfc',
       'snowfall_amount_sfc', 'total_cloud_cover_sfc',
       'high_cloud_cover_high_cld_lay', 'medium_cloud_cover_mid_cld_lay',
       'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc',
       'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd',
       'wind_speed_900_mb', 'wind_direction_900_mb',
       'wind_gust_10_m_above_gnd', 'angle_of_incidence','azimuth',
       'generated_power_kw'],axis="columns")

print(df_to_be_predicted.columns)

features = pd.DataFrame(scaler.fit_transform(df_to_be_predicted), columns = df_to_be_predicted.columns)
y_predicted = clf.predict(features)

#DHI_values = pd.DataFrame(y_predicted)
#DHI_values.rename(columns={0:'DHI'}, inplace=True)
#print(DHI_values)
#DHI_values.to_csv('DHI_predicted.csv',index=False)
#plt.hist(DHI_values["DHI"],bins=50,rwidth=0.8)
#plt.show()





#plt.scatter(df2['DNI'], df2['Hour'], marker='+')
#plt.xlabel('DNI')
#plt.ylabel('Hour')
#plt.show()

#plt.scatter(df2['GHI'], df2['Temperature(deg.C)'], marker='+')
#plt.xlabel('GHI')
#plt.ylabel('Temperature(deg.C)')
#plt.show()

df2['GHI/temp'] = df2['GHI']/df2['Temperature(deg.C)']
#plt.hist(df2['GHI/temp'],rwidth=0.8,bins=50)
#plt.show()

df2 = df2[df2['GHI/temp']<30]
df2 = df2.drop(['GHI/temp'],axis='columns')

#plt.scatter(df2['GHI'], df2['Relative humidity(%)'], marker='+')
#plt.xlabel('GHI')
#plt.ylabel('Relative humidity(%)')
#plt.show()

df2['GHI/Relative humidity'] = df2['GHI']/df2['Relative humidity(%)']
#plt.hist(df2['GHI/Relative humidity'],rwidth=0.8,bins=50)
#plt.show()

df2 = df2[df2['GHI/Relative humidity']<10]

df2 = df2.drop(['GHI/Relative humidity'],axis='columns')

#plt.scatter(df2['GHI'], df2['Pressure(mbar)'], marker='+')
#plt.xlabel('GHI')
#plt.ylabel('Pressure(mbar)')
#plt.show()

df2['GHI/pressure'] = df2['GHI']/df2['Pressure(mbar)']
#plt.hist(df2['GHI/pressure'],rwidth=0.8,bins=50)
#plt.show()

df2 = df2.drop(['GHI/pressure'],axis='columns')

#plt.scatter(df2['DNI'], df2['Wind Speed(m/s)'], marker='+')
#plt.xlabel('DNI')
#plt.ylabel('Wind Direction(degree)')
#plt.show()

scaler = StandardScaler()

# Removing the target columns
df2_target = df2.drop(['Hour', 'Temperature(deg.C)','Relative humidity(%)', 'Pressure(mbar)', 'Wind Direction(degree)','Wind Speed(m/s)','Zenith(degree)'],axis='columns')
df2_features = df2.drop(['DHI(W/m2)', 'DNI', 'EBH', 'GHI','Hour'],axis='columns')
print(df2_features.columns)
df2_features = pd.DataFrame(scaler.fit_transform(df2_features), columns = df2_features.columns)
y = df2_target['GHI']

print(df2_features.shape)
print(X_train.shape,X_test.shape)

X_train, X_test, y_train, y_test = train_test_split(df2_features, y, test_size=0.3, random_state=101)
clf = RandomForestRegressor(n_estimators=100,criterion='mse',max_features='auto')
clf.fit(X_train,y_train)
#with open("Auxiliary_model_GHI.pickle",'wb') as f:
#       pickle.dump(clf,f)
print(clf.score(X_test,y_test)) # 0.8630321081437395
#lpredictions = clf.predict(X_test)
#print(lpredictions)
#print(y_test)


df_to_be_predicted2 = pd.read_csv('seen.csv')

df_to_be_predicted2 = df_to_be_predicted2.drop(['total_precipitation_sfc',
       'snowfall_amount_sfc', 'total_cloud_cover_sfc',
       'high_cloud_cover_high_cld_lay', 'medium_cloud_cover_mid_cld_lay',
       'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc',
       'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd',
       'wind_speed_900_mb', 'wind_direction_900_mb',
       'wind_gust_10_m_above_gnd', 'angle_of_incidence','azimuth',
       'generated_power_kw'],axis="columns")

print(df_to_be_predicted2.columns)

features = pd.DataFrame(scaler.fit_transform(df_to_be_predicted2), columns = df_to_be_predicted2.columns)

#y_predicted = clf.predict(features)

#GHI_values = pd.DataFrame(y_predicted)
#GHI_values = GHI_values.rename(columns={0:'GHI'})
#GHI_values.to_csv('GHI_predicted.csv',index=False)
#plt.hist(GHI_values['GHI'],bins=50,rwidth=0.8)
#plt.show()

columns = {
       'data_columns' : [col for col in df2_features.columns]
}
#print()
with open('columns.json','w') as f:
       f.write(json.dumps(columns))


































