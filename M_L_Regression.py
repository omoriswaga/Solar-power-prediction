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
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#style.use('ggplot')

SEQ_LEN = 80
FUTURE_PERIOD_PREDICT = 6
EPOCHS = 10
BATCH_SIZE = 64

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

df = pd.read_csv("readings.csv")
df = df.dropna()
df = df.drop(['Day of Year','Year','Month','Day','First Hour of Period'], axis = 'columns')
le = LabelEncoder()
df['Is Daylight'] = le.fit_transform(df['Is Daylight'])
forecast_col = 'Power Generated'

forecast_out = int(math.ceil(0.004*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.isna().sum())
X = np.array(df.drop(['label'],axis = 'columns'))
X  = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df = df.dropna()
y = np.array(df['label'])


kf = KFold(n_splits = 10)
#for train_index, test_index in kf.split(X):
#    X_train,X_test,y_train,y_test = X[train_index],X[test_index],y[train_index], y[test_index]
#    clf = LinearRegression()
#    clf.fit(X_train,y_train)
#    print(clf.score(X_test,y_test))



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
scores = []
cv = ShuffleSplit(n_splits = 10, test_size=0.2, random_state = 0)
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

#<class 'dict'>: {'criterion': 'mse', 'max_features': 'auto', 'n_estimators': 100}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=10)
clf = RandomForestRegressor(criterion = 'mse', max_features='auto', n_estimators=100)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

forecast_set = clf.predict(X_lately)
print(forecast_set)

print(df)

df['Forcast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
print(df['Power Generated'])
df['Power Generated'].plot(grid=True)
df['Forcast'].plot(grid=True)
plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Power Generated')
plt.show()
print(int(434.10095))

































import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
df = pd.read_csv('seen.csv')
predicted_DHI = pd.read_csv("DHI_predicted.csv")
predicted_GHI = pd.read_csv("GHI_predicted.csv")
print(len(predicted_DHI),len(df))
df = pd.concat([df,prediction_df],axis='columns')
df = df.rename(columns={'0':'Radiation'})
df = df.dropna()
le = LabelEncoder()
df['Is Daylight'] = le.fit_transform(df['Is Daylight'])
df = df.drop(['Day of Year','Day','Sky Cover','Visibility','Month','Year'], axis = 'columns')
print(df)
forecast_out = int(math.ceil(0.006*len(df)))
print(forecast_out)

plt.scatter(df['Radiation'],df['Power Generated'])
plt.show()
_X = df[:-forecast_out]
X_lately = df[-forecast_out:]

pd.to_numeric(df['Power Generated'],errors='coerce')
df['Power Generated'] = df['Power Generated'].apply(lambda x: float(x))

plt.hist(df['Power Generated'],rwidth=0.8,bins=50)
#plt.show()
#df = df[df['generated_power_kw']<2500]
print(df.shape)
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
_X1 = _X.drop(['Power Generated'],axis='columns')
sc = MinMaxScaler()
_X1 = sc.fit_transform(_X1)
y_data = sc.fit_transform(_X[['Power Generated']])
X,y = lstm_data_transform(_X1,y_data)
print(len(X),len(y))
_X_test = X_lately.drop(['Power Generated'],axis='columns')
_X_test = sc.fit_transform(_X_test)
_y_test = sc.fit_transform(X_lately[['Power Generated']])
X_test,y_test = lstm_data_transform(_X_test,_y_test)
print(len(X_test),len(y_test))
#X_shaped = np.reshape(_X,newshape=(-1,5,3794))


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

#y_pred = classifier.predict(X_test)
#print(y_pred)
#print(y_test)

# 2 dense 32 layer 200 9.38e-3
# 5 dense 32 layer 250 0.013
# 3 dense 120 layers 7.91e-3


NAME = "real 3-dense_layers,120-layer_sizes Solar_prediction_neural_network-{}".format(int(time.time()))
tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
classifier = Sequential()
classifier.add(LSTM(120, activation='relu',input_shape=(X.shape[1:]), return_sequences=False))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())

classifier.add(Dense(units=120, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units=120, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units=120, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())

classifier.add(Dense(units=1,activation='linear'))

opt = tf.keras.optimizers.Adam(lr = 0.001, decay=1e-6)
classifier.compile(loss = 'mse', optimizer = opt, metrics=['mae','accuracy'])
#classifier.fit(X,y, batch_size = 30, epochs = 250, validation_split=0.2, callbacks=[tensor_board])

#model_json = classifier.to_json()
#with open("Main_model_3D.json",'w') as json_file:
#    json_file.write(model_json)
#classifier.save_weights("main_model_3D.h5")
#print("Saved model to disk")

# 2 dense 32 layer 200 0.0114
# 5 dense 32 layers 250 0.0133
# 3 dense 120 layers 250 0.0117


# load json and create model

json_file = open('Main_model_5D.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model

loaded_model.load_weights('main_model_5D.h5')
print('Loaded model from disk')

# Evaluate model

loaded_model.compile(loss = 'mse', optimizer = opt, metrics=['accuracy'])
score = loaded_model.evaluate(X_test,y_test,verbose=0)
print(score)
y_pred = loaded_model.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
for _ in y_pred:
    print(_)

print("Real values")
y_test = sc.inverse_transform(y_test)
for _ in y_test:
    print(_)




