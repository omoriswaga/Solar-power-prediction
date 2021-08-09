from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import json
import pickle
import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math

__data_columns = None
__auxiliary_model_DHI = None
__auxiliary_model_GHI = None
__Main_model = None
__predicted_GHI = None
__predicted_DHI = None
df = None
df_y = None
df2 = None
percentage_out = None


app = Flask(__name__)

@app.route('/get_location_structure')
def get_location_structure():
    response = jsonify ({
        'columns' : load_saved_artifacts()
    })
    response.headers.add('Acess-Control-Allow-origin', '*')
    return response


def get_estimated_DHI(Temperature, Relative_humidity, Pressure, Wind_Direction, Wind_Speed, Zenith):
    global __auxiliary_model_DHI
    global __predicted_DHI
    try:
        x = np.zeros(len(__data_columns))
        x[0] = Temperature
        x[1] = Relative_humidity
        x[2] = Pressure
        x[3] = Wind_Direction
        x[4] = Wind_Speed
        x[5] = Zenith

        scaler = StandardScaler()
        x = x.reshape(-1, 1)
        x = scaler.fit_transform(x)
        x = np.reshape(x, (1, 6))

        __predicted_DHI =  __auxiliary_model_DHI.predict(x)
        return __predicted_DHI
    except Exception as e:
        print(e)
def lstm_data_transform(x_data,num_step = 11):
    X= list()
    for i in range(x_data.shape[0]):
        end_ix = i + num_step
        if end_ix > x_data.shape[0]:
            break
        seq_x = x_data[i:end_ix]
        X.append(seq_x)
    x_array = np.array(X)
    return x_array

def get_estimated_power_output_kw(temperature_2_m_above_gnd, relative_humidity_2_m_above_gnd,
       mean_sea_level_pressure_MSL, wind_direction_10_m_above_gnd,
       wind_speed_10_m_above_gnd, zenith, azimuth,DHI,GHI):
    global __Main_model
    global __predicted_DHI
    global __predicted_GHI
    global df
    global df_y
    global percentage_out
    try:
        x = np.zeros(9)
        x[0] = temperature_2_m_above_gnd
        x[1] = relative_humidity_2_m_above_gnd
        x[2] = mean_sea_level_pressure_MSL
        x[3] = wind_direction_10_m_above_gnd
        x[4] = wind_speed_10_m_above_gnd
        x[5] = zenith
        x[6] = azimuth
        x[7] = DHI
        x[8] = GHI
        out = percentage_out.append(pd.Series([temperature_2_m_above_gnd, relative_humidity_2_m_above_gnd,mean_sea_level_pressure_MSL, wind_direction_10_m_above_gnd,
        wind_speed_10_m_above_gnd, zenith, azimuth,DHI,GHI],index=percentage_out.columns),ignore_index=True)
        #print(percentage_out)
        sc = MinMaxScaler()
        sc_y = MinMaxScaler()
        sc_y.fit(df_y)
        sc.fit(df)
        #x = x.reshape(1, -1)
        #x = pd.DataFrame(x)
        #print(x)
        x = sc.transform(out)
        #x = np.reshape(x, (1, 9))
        x = lstm_data_transform(x)
        y = __Main_model.predict(x)
        y = sc_y.inverse_transform(y)
        print(y)
        return y
    except Exception as e:
        print(e)


def get_estimated_GHI(Temperature, Relative_humidity, Pressure, Wind_Direction, Wind_Speed, Zenith):
    global __auxiliary_model_GHI
    global __predicted_GHI
    global df2
    try:
        x = np.zeros(len(__data_columns))
        x[0] = Temperature
        x[1] = Relative_humidity
        x[2] = Pressure
        x[3] = Wind_Direction
        x[4] = Wind_Speed
        x[5] = Zenith
        scaler = StandardScaler()
        x = x.reshape(-1, 1)
        x = scaler.fit_transform(x)
        x = np.reshape(x, (1, 6))
        __predicted_GHI = __auxiliary_model_GHI.predict(x)
        return __predicted_GHI

    except Exception as e:
        print(e)


def load_saved_artifacts():
    print("Loading saved artifacts......")
    global __data_columns
    global  __auxiliary_model_DHI
    global  __auxiliary_model_GHI
    global __Main_model
    global df
    global df_y
    global df2
    global percentage_out

    with open('columns.json','r') as f:
        __data_columns = json.load(f)['data_columns']

    with open('Auxiliary_model_DHI.pickle','rb') as f:
        __auxiliary_model_DHI = pickle.load(f)

    with open('Auxiliary_model_GHI.pickle','rb') as f:
        __auxiliary_model_GHI = pickle.load(f)

    json_file = open('Main_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    __Main_model = model_from_json(loaded_model_json)
    __Main_model.load_weights('main_model_weight.h5')

    df = pd.read_csv('seen.csv')
    predicted_DHI = pd.read_csv("DHI_predicted.csv")
    predicted_GHI = pd.read_csv("GHI_predicted.csv")
    df = pd.concat([df, predicted_DHI, predicted_GHI], axis='columns')
    df = df.dropna()
    df_y = df[['generated_power_kw']]
    df = df.drop(
        ['total_precipitation_sfc', 'snowfall_amount_sfc', 'total_cloud_cover_sfc', 'high_cloud_cover_high_cld_lay',
         'medium_cloud_cover_mid_cld_lay', 'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc',
         'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd', 'wind_speed_900_mb', 'wind_direction_900_mb',
         'wind_gust_10_m_above_gnd', 'angle_of_incidence','generated_power_kw'], axis='columns')

    df2 = pd.read_csv("6.401852_5.615612_SAM_p50.csv")
    df2 = df2.dropna()
    df2 = df2.drop(['Year', 'Month', 'Day', 'Minute', 'Albedo', 'Azimuth(degree)', 'cloudopacity(%)', 'Dew Point(deg.c)',
                  'Snow Depth(cm)'], axis='columns')
    df2 = df2.drop(['DHI(W/m2)', 'DNI', 'EBH', 'GHI', 'Hour'], axis='columns')

    percentage_out = int(math.ceil(0.00237*len(df)))
    percentage_out = df[-percentage_out:]

    print("Loading saved artifacts done....")
    return __data_columns



if __name__ == '__main__':
    print("Starting python Flask server for solar power output prediction")
    load_saved_artifacts()
    #get_estimated_GHI(23.6, 78.7, 1000.8, 270, 1.1, 68)
    #get_estimated_DHI(23.6, 78.7, 1000.8, 270, 1.1, 68)
    get_estimated_power_output_kw(8, 80, 1025.3, 12.72, 11.4, 76.9195, 223.049, 0.24, 0)
    app.run()
    get_estimated_power_output_kw(8, 80, 1025.3, 12.72, 11.4, 76.9195,223.049,0.24,0)
