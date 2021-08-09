import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import deque
import random
import math
from sklearn.metrics import mean_squared_error
from csv import reader
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import KFold, train_test_split, ShuffleSplit,GridSearchCV
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
content = []
with open("solarpowergeneration.csv") as file:
    csv_reader = reader(file)
    for row in csv_reader:
        for _ in row:
            row = _.split(";")
        content.append(row)
df = pd.DataFrame(content,columns=['temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd', 'mean_sea_level_pressure_MSL', 'total_precipitation_sfc', 'snowfall_amount_sfc', 'total_cloud_cover_sfc', 'high_cloud_cover_high_cld_lay', 'medium_cloud_cover_mid_cld_lay', 'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc', 'wind_speed_10_m_above_gnd', 'wind_direction_10_m_above_gnd', 'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd', 'wind_speed_900_mb', 'wind_direction_900_mb', 'wind_gust_10_m_above_gnd', 'angle_of_incidence', 'zenith', 'azimuth', 'generated_power_kw'])
df = df[df['temperature_2_m_above_gnd']!='temperature_2_m_above_gnd']
print(df)
print(df.columns)
df = df.dropna()
df = df.drop(['snowfall_amount_sfc'], axis = 'columns')

forecast_col = 'generated_power_kw'

forecast_out = int(math.ceil(0.002*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.isna().sum())
X = np.array(df.drop(['label'],axis = 'columns'))
X  = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df = df.dropna()
y = np.array(df['label'])

model_params = {
    'svm': {
        'model': svm.SVR(gamma='auto'),
        'params': {
            'C': [10,20,40],
            'kernel': ['rbf','linear']
        }
    },
    'linear_regression': {
        'model': LinearRegression(),
        'params': {
            'normalize': [True,False]
        }
    },
    'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [1,2],
            'selection': ['random','cyclic']
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse'],
            'splitter': ['best','random']
        }
    },
    'random_forest' : {
        'model' : RandomForestRegressor(),
        'params': {
            'n_estimators': [10,30,50,100],
            'criterion' : ['mse', 'friedman_mse'],
            'max_features': ["auto", "sqrt", "log2"]
        }
    }

}
#scores = []
#cv = ShuffleSplit(n_splits = 10, test_size=0.2, random_state = 0)
#for aldo_name, config in model_params.items():
#    gs = GridSearchCV(config['model'],config['params'], cv=cv, return_train_score = False)
#    gs.fit(X,y)
#    scores.append({
#        'model': aldo_name,
#        'best_score': gs.best_score_,
#        'best_params': gs.best_params_
#    })
#output = pd.DataFrame(scores,columns=['model','best_score','best_params'])
#print(output)
#print(len(X),len(y))

#0                svm  ...                         {'C': 40, 'kernel': 'rbf'}
#1  linear_regression  ...                                {'normalize': True}
#2              lasso  ...                {'alpha': 2, 'selection': 'cyclic'}
#3      decision_tree  ...  {'criterion': 'friedman_mse', 'splitter': 'best'}
#4      random_forest  ...  {'criterion': 'mse', 'max_features': 'auto', '...

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=10)
clf = RandomForestRegressor(criterion = 'mse', max_features='auto', n_estimators=200)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

forecast_set = clf.predict(X_lately)
print(forecast_set)

df['Forcast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
pd.to_numeric(df['generated_power_kw'],errors='coerce')
pd.to_numeric(df['Forcast'],errors='coerce')
df['generated_power_kw'] = df['generated_power_kw'].apply(lambda x: float(x))
df['Forcast'] = df['Forcast'].apply(lambda x: float(x))
#print(df['generated_power_kw'])
df['generated_power_kw'].plot(grid=True)
df['Forcast'].plot(grid=True)
plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Power Generated')
plt.show()























df = pd.read_csv("SolarPrediction.csv")

#print(df.head())

df['Hour'] = df['Time'].apply(lambda x: x.split(':')[0])
pd.to_numeric(df['Hour'],errors='coerce')
df['Hour'] = df['Hour'].apply(lambda x: int(x))

#checking relationship between radiation and temperature

#plt.scatter(df['Radiation'], df['Temperature'], marker='+')
#plt.xlabel('Radiation')
#plt.ylabel('Temperature')
#plt.show()

#plt.scatter(df['Radiation'], df['Humidity'], marker='*')
#plt.xlabel('Radiation')
#plt.ylabel('Humidity')
#plt.show()

#plt.scatter(df['Radiation'],df['Hour'], marker='+')
#plt.ylabel('Hour')
#plt.xlabel('Radiation')
#plt.show()

#plt.scatter( df['Radiation'],df['Speed'], marker='+')
#plt.ylabel('Speed')
#plt.xlabel('Radiation')
#plt.show()

plt.scatter(df['Radiation'],df['Pressure'], marker='+')
plt.ylabel('Pressure')
plt.xlabel('Radiation')
plt.show()

#plt.hist(df['Radiation'],bins=50,rwidth=0.8)
#plt.show()

# Outlier removal

#drop low radiation values
df = df[df['Radiation'] >= 10]

df['Radiation/temp'] = df['Radiation']/df['Temperature']
#plt.hist(df['Radiation/temp'],rwidth=0.8,bins=50)
#plt.show()
print(len(df[df['Radiation/temp']>17]))
df = df[df['Radiation/temp']<17]
#plt.scatter(df['Radiation'], df['Temperature'], marker='+')
#plt.xlabel('Radiation')
#plt.ylabel('Temperature')
#plt.show()
df = df.drop(['Radiation/temp'],axis='columns')

df['speed/Radiation'] = df["Speed"]/df["Radiation"]
#plt.hist(df['speed/Radiation'],rwidth=0.8,bins=50)
#plt.show()
df = df[df['speed/Radiation']<0.5]
#plt.scatter( df['Radiation'],df['Speed'], marker='+')
#plt.ylabel('Speed')
#plt.xlabel('Radiation')
#plt.show()
df = df.drop(['speed/Radiation'],axis='columns')

df['Radiation/pressure'] = df['Radiation']/df['Pressure']
#plt.hist(df['Radiation/pressure'],rwidth=0.8,bins=50)
#plt.show()
df = df[df['Radiation/pressure']<35]
df = df.drop(['Radiation/pressure'],axis='columns')





#Feature Enginering
#df['Time_conv'] =  pd.to_datetime(df['Time'], format='%H:%M:%S')
#df['hour'] = pd.to_datetime(df['Time_conv'], format='%H:%M:%S').dt.hour

#Add column 'month'
df['month'] = pd.to_datetime(df['UNIXTime'].astype(int), unit='s').dt.month

#Add column 'year'
df['year'] = pd.to_datetime(df['UNIXTime'].astype(int), unit='s').dt.year

#Duration of Day
df['total_time'] = pd.to_datetime(df['TimeSunSet'], format='%H:%M:%S').dt.hour - pd.to_datetime(df['TimeSunRise'], format='%H:%M:%S').dt.hour
#print(df.head())
#Data visualization
#ax = plt.axes()
#sns.barplot(x="month", y='Radiation', data = df, palette="BuPu", ax = ax, order=[9,10,11,12,1])
#ax.set_title('Mean Radiation by Month')
#plt.show()


#ax = plt.axes()
#sns.barplot(x="total_time", y='Radiation', data=df, palette="BuPu", ax = ax)
#ax.set_title('Radiation by Total Daylight Hours')
#plt.show()


df = df.dropna()
y = df.Radiation
X = df.drop(['Radiation','Data', 'Time', 'TimeSunRise', 'TimeSunSet','total_time','UNIXTime','WindDirection(Degrees)',"month",'year','Humidity'], axis='columns')
print(X.columns)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

model_params = {
    'svm': {
        'model': svm.SVR(gamma='auto'),
        'params': {
            'C': [10,20,40,70],
            'kernel': ['rbf','linear']
        }
    },
    'linear_regression': {
        'model': LinearRegression(),
        'params': {
            'normalize': [True,False]
        }
    },
    'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [1,2],
            'selection': ['random','cyclic']
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['mse', 'friedman_mse'],
            'splitter': ['best','random']
        }
    },
    'random_forest' : {
        'model' : RandomForestRegressor(),
        'params': {
            'n_estimators': [10,30,50,100],
            'criterion' : ['mse', 'friedman_mse'],
            'max_features': ["auto", "sqrt", "log2"]
        }
    }

}
#scores = []
#cv = ShuffleSplit(n_splits = 10, test_size=0.2, random_state = 0)
#for aldo_name, config in model_params.items():
#    gs = GridSearchCV(config['model'],config['params'], cv=cv, return_train_score = False)
#    gs.fit(X,y)
#    scores.append({
#        'model': aldo_name,
#        'best_score': gs.best_score_,
#        'best_params': gs.best_params_
#    })
#output = pd.DataFrame(scores,columns=['model','best_score','best_params'])
#print(output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
clf = RandomForestRegressor(n_estimators=100,criterion='mse',max_features='sqrt')
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
lpredictions = clf.predict(X_test)
print(lpredictions)
print(y_test)

#print('MAE:', metrics.mean_absolute_error(y_test, lpredictions))
#print('MSE:', metrics.mean_squared_error(y_test, lpredictions))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lpredictions)))

# predicting radiation from our weather forcast
df2 = pd.read_csv("readings.csv")
dt = datetime.date(2019,10,19)
unix_time = []
for i in range(len(df2)):
    unix_time.append(int(time.mktime(datetime.date(df2.Year.values[i],df2.Month.values[i],df2.Day.values[i]).timetuple())))
#df2['UNIXTime'] = time.mktime(datetime.date(df2.Year,df2.Month,df2.Day).timetuple())
df2 = df2.drop(['Day of Year','Day','Distance to Solar Noon','Sky Cover','Visibility','Average Wind Speed (Period)','Power Generated','Is Daylight','Average Wind Direction (Day)','Month',"Year",'Relative Humidity'],axis='columns')
unix_time_df = pd.DataFrame(unix_time,columns=['UNIXTime'])
#df2['Average Wind Speed (Day)'] = df2['Average Wind Speed (Day)'].apply(lambda x: x*0.621)
#print(unix_time_df)
#df2 = pd.concat([unix_time_df,df2],axis="columns")
print(df2.columns)
#print(df2['Average Wind Speed (Day)'])
X_predict_data = pd.DataFrame(scaler.fit_transform(df2), columns = df2.columns)
y_predicted = clf.predict(X_predict_data)

df3 = pd.DataFrame(y_predicted)
print(df3.head())
df3.to_csv('predicted_radiation.csv',index=False)
plt.hist(df3[0],bins=50,rwidth=0.8)
plt.show()



















