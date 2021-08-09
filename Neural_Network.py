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
import time
from tensorflow.keras.callbacks import TensorBoard
import pickle
from keras.models import model_from_json
from keras import regularizers

#content = []
#with open("solarpowergeneration.csv") as file:
#    csv_reader = reader(file)
#    for row in csv_reader:
#        for _ in row:
#            row = _.split(";")
#        content.append(row)
#df = pd.DataFrame(content,columns=['temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd', 'mean_sea_level_pressure_MSL', 'total_precipitation_sfc', 'snowfall_amount_sfc', 'total_cloud_cover_sfc', 'high_cloud_cover_high_cld_lay', 'medium_cloud_cover_mid_cld_lay', 'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc', 'wind_speed_10_m_above_gnd', 'wind_direction_10_m_above_gnd', 'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd', 'wind_speed_900_mb', 'wind_direction_900_mb', 'wind_gust_10_m_above_gnd', 'angle_of_incidence', 'zenith', 'azimuth', 'generated_power_kw'])
#df = df[df['temperature_2_m_above_gnd']!='temperature_2_m_above_gnd']
#df = df.dropna()
#df = df.drop(['snowfall_amount_sfc'], axis = 'columns')
df = pd.read_csv('seen.csv') #df=> data frame for short
predicted_DHI = pd.read_csv("DHI_predicted.csv")
predicted_GHI = pd.read_csv("GHI_predicted.csv")
#print(len(predicted_DHI),len(df))
df = pd.concat([df,predicted_DHI,predicted_GHI],axis='columns')
df = df.dropna()
df = df.drop(['total_precipitation_sfc','snowfall_amount_sfc','total_cloud_cover_sfc','high_cloud_cover_high_cld_lay','medium_cloud_cover_mid_cld_lay','low_cloud_cover_low_cld_lay','shortwave_radiation_backwards_sfc','wind_speed_80_m_above_gnd','wind_direction_80_m_above_gnd','wind_speed_900_mb','wind_direction_900_mb','wind_gust_10_m_above_gnd','angle_of_incidence'],axis='columns')
print(df.columns)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

# Visualizing datasets

plt.hist(df['generated_power_kw'],bins=50,rwidth=0.8)
plt.xlabel("range")
plt.ylabel("generated_power_kw")
plt.show()

plt.hist(df['temperature_2_m_above_gnd'],bins=50,rwidth=0.8)
plt.xlabel("range")
plt.ylabel("temperature_2_m_above_gnd")
plt.show()

plt.hist(df['relative_humidity_2_m_above_gnd'],bins=50,rwidth=0.8)
plt.xlabel("range")
plt.ylabel("relative_humidity_2_m_above_gnd")
plt.show()

plt.hist(df['mean_sea_level_pressure_MSL'],bins=50,rwidth=0.8)
plt.xlabel("range")
plt.ylabel("mean_sea_level_pressure_MSL")
plt.show()
print(df.shape)

Q3 = df['mean_sea_level_pressure_MSL'].quantile(0.75)
Q1 = df['mean_sea_level_pressure_MSL'].quantile(0.25)

IQR = Q3-Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

df = df[(df['mean_sea_level_pressure_MSL']>lower_limit) & (df['mean_sea_level_pressure_MSL']<upper_limit)]
print(df.shape)

#plt.scatter(df['generated_power_kw'], df['temperature_2_m_above_gnd'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('temperature_2_m_above_gnd')
#plt.show()

#plt.scatter(df['generated_power_kw'], df['relative_humidity_2_m_above_gnd'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('relative_humidity_2_m_above_gnd')
#plt.show()

#plt.scatter(df['generated_power_kw'], df['mean_sea_level_pressure_MSL'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('mean_sea_level_pressure_MSL')
#plt.show()

#plt.scatter(df['generated_power_kw'], df['wind_direction_10_m_above_gnd'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('wind_direction_10_m_above_gnd')
#plt.show()

#plt.scatter(df['generated_power_kw'], df['wind_speed_10_m_above_gnd'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('wind_speed_10_m_above_gnd')
#plt.show()

#plt.scatter(df['generated_power_kw'], df['zenith'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('zenith')
#plt.show()

#plt.scatter(df['generated_power_kw'], df['azimuth'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('azimuth')
#plt.show()

#plt.scatter(df['generated_power_kw'], df['DHI'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('DHI')
#plt.show()

#plt.scatter(df['generated_power_kw'], df['GHI'], marker='+')
#plt.xlabel('generated_power_kw')
#plt.ylabel('GHI')
#plt.show()


_X = df[:-forecast_out]
X_lately = df[-forecast_out:]
y = df[['generated_power_kw']]
_y = y[:-forecast_out]
y_lately = y[-forecast_out:]

print(df.shape, df.columns)
print(_X.shape,_y.shape)

def lstm_data_transform(x_data,y_data,num_step = 11):
    X,y= list(),list()
    for i in range(x_data.shape[0]):
        end_ix = i + num_step
        if end_ix >= x_data.shape[0]:
            break
        seq_x = x_data[i:end_ix]
        seq_y = y_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array , y_array

_X1 = _X.drop(['generated_power_kw'],axis='columns')
sc_x = MinMaxScaler()
sc_y = MinMaxScaler()

_X1 = sc_x.fit_transform(_X1)
y_data = sc_y.fit_transform(_y)
X,y = lstm_data_transform(_X1,y_data)
print(len(X),len(y))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)
print(X_train.shape,X_test.shape)

X_forcast = X_lately.drop(["generated_power_kw"], axis = 'columns')
X_forcast = sc_x.transform(X_forcast)
y_forcast = sc_y.transform(y_lately)

X_for,y_for = lstm_data_transform(X_forcast,y_forcast)

#dense_layers = [2,3,5]
#layer_sizes = [32,64,120,200]

#for dense_layer in dense_layers:
#    for layer_size in layer_sizes:
#        NAME = "{}-dense_layers,{}-layer_sizes Solar_prediction_neural_network-{}".format(dense_layer,layer_size,int(time.time()))
#        tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
#        classifier = Sequential()
#        classifier.add(LSTM(layer_size, activation='relu',input_shape=(X.shape[1:]), return_sequences=False))
#        classifier.add(Dropout(0.2))
#        classifier.add(BatchNormalization())

#        for l in range(dense_layer):
#            classifier.add(Dense(units=layer_size, kernel_initializer='uniform', activation='relu'))
#            classifier.add(Dropout(0.2))
#            classifier.add(BatchNormalization())

#        classifier.add(Dense(units=1,activation='linear'))

#        opt = tf.keras.optimizers.Adam(lr = 0.001, decay=1e-6)
#        classifier.compile(loss = 'mse', optimizer = opt, metrics=['mae'])

#        history = classifier.fit(X,y, batch_size = 30, epochs = 200, validation_split=0.2, callbacks=[tensor_board])


NAME = "real 1-dense_layers,5-layer_sizes Solar_prediction_neural_network-{}".format(int(time.time()))
tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
classifier = Sequential()
classifier.add(LSTM(4, activation='relu',input_shape=(X_train.shape[1:]),return_sequences=True))
classifier.add(BatchNormalization())

classifier.add(LSTM(4, activation='relu',input_shape=(X_train.shape[1:]),return_sequences=False))
classifier.add(BatchNormalization())

classifier.add(Dense(units=32, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.001), activation='relu'))

classifier.add(Dense(units=32, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.001), activation='relu'))

classifier.add(Dense(units=1,activation='linear'))

opt = tf.keras.optimizers.Adam(lr = 0.01, decay=1e-5)
#classifier.compile(loss = 'mse', optimizer = opt, metrics=['mae'])
#classifier.fit(X_train,y_train, batch_size = 50, epochs = 100, validation_split=0.2, callbacks=[tensor_board])

#model_json = classifier.to_json()
#with open("Main_model.json",'w') as json_file:
#    json_file.write(model_json)
#classifier.save_weights("main_model_weight.h5")
#print("Saved model to disk")

# load json and create model

json_file = open('Main_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights('main_model_weight.h5')
print('Loaded model from disk')

loaded_model.compile(loss = 'mae', optimizer = opt, metrics=['mae'])
score = loaded_model.evaluate(X_test,y_test,verbose=0)
print(score)
pred = loaded_model.predict(X_test)

range_list = []
for i in range(len(y_test)):
    range_list.append(i)
plt.plot(range_list,y_test,label='actual values')
plt.plot(range_list,pred,label='predicted values')
plt.xlabel('Range of values')
plt.ylabel("Generated output power in KW")
plt.legend()
plt.show()


y_pred = loaded_model.predict(X_for)
y_pred = sc_y.inverse_transform(y_pred)
for _ in y_pred:
    print(_)

print("Real values")
y_test = sc_y.inverse_transform(y_for)
for _ in y_test:
    print(_)


df['Forcast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date
one_day = 86400
next_unix = last_unix + one_day
for i in y_pred:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
df = df.astype(float)
df['generated_power_kw'].plot(grid=True)
df['Forcast'].plot(grid=True)
plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Power Generated')
plt.show()


