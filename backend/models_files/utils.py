import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pathlib as Path
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def scalingfortrain(x_inp , scaled_columns , scaler , unimportant_cols_todrop):
    if unimportant_cols_todrop is None:
        x_scaled = scaler.fit_transform( x_inp[scaled_columns] )
        x_nonscaled = x_inp.drop(columns=scaled_columns)
        final_x = np.concatenate((x_scaled ,x_nonscaled) ,axis=1)
    else:
        x_scaled = scaler.fit_transform(x_inp[scaled_columns])
        x_nonscaled = x_inp.drop(columns=scaled_columns + unimportant_cols_todrop)
        final_x = np.concatenate((x_scaled ,x_nonscaled) ,axis=1)
    return final_x
def scalingfortest(x_inp , scaled_columns , scaler , unimportant_cols_todrop):
    if unimportant_cols_todrop is None:
        x_scaled = scaler.transform(x_inp[scaled_columns])
        x_nonscaled = x_inp.drop(columns=scaled_columns)
        final_x = np.concatenate((x_scaled ,x_nonscaled) ,axis=1)
    else:
        x_scaled = scaler.transform(x_inp[scaled_columns])
        x_nonscaled = x_inp.drop(columns=scaled_columns + unimportant_cols_todrop)
        final_x = np.concatenate((x_scaled ,x_nonscaled) ,axis=1)
    return final_x

def data_splitting(X ,Y ,train_size ,test_size ,valid_size ,random_state):
    no_1 = test_size + valid_size
    no_2 = test_size / (test_size + valid_size)
    x_train ,x_temp ,y_train ,y_temp = train_test_split(
        X ,Y ,
        test_size=no_1 ,
        random_state=random_state
    )
    x_valid ,x_test ,y_valid ,y_test = train_test_split(
        x_temp ,y_temp ,
        test_size=no_2 ,
        random_state=random_state
    )
    return x_train ,x_valid ,x_test ,y_train ,y_valid ,y_test

def replace_values_in_csv(df ,column_name ,value_0 ,value_1 ,value_2):
    if value_2 is None:
        df[column_name] = df[column_name].replace({value_0: 0,value_1: 1}).astype(int)
    else:
        df[column_name] = df[column_name].replace({value_0:0 ,value_1: 1,value_2: 2}).astype(int)
    return df[column_name]

def prepare_input(values: dict, feature_order: list):
    """
    values:        dict جاي من الفرونت إند، keys = أسماء الأعمدة
    feature_order: ليستة بترتيب الأعمدة بالظبط زي وقت التدريب (X.columns)

    بترجع: numpy array بشكل (1, n_features) جاهز يدخل على الـ scaler/model
    """
    try:
        row = [float(values[col]) for col in feature_order]
    except KeyError as missing:
        raise ValueError(f"Missing required field: {missing}") from missing

    return np.array([row])

def MAP(systolic_bp ,diastolic_bp):
    return  float(diastolic_bp) + (1/3) * (float(systolic_bp) - float(diastolic_bp))

def RPP(systolic_bp ,heart_rate):
    return float(systolic_bp) * float(heart_rate)

def PP(systolic_bp ,diastolic_bp):
    return float(systolic_bp) - float(diastolic_bp)

def UnhealthyLifeScore(smoking_status ,alcohol_consumption ,physical_activity):
    score = 0
    if float(smoking_status) == 1:
        score += 1
    if float(alcohol_consumption) == 1:
        score += 1
    if float(physical_activity) == 0:
        score += 1
    elif float(physical_activity) == 1:
        score += 0.05
    elif float(physical_activity) == 2:
        score += 0
    return score

def AtherogenicIndexCoefficient(cholestrol_mg_dl ,systolic_bp):
    return float(cholestrol_mg_dl) * float(systolic_bp)

def SmokingHypertensionInteraction(smoking_status ,systolic_bp):
    if float(smoking_status) == 1:
        return float(systolic_bp) * 1.2
    return float(systolic_bp)

def CardiacAdiposityProxy(bmi ,heart_rate):
    return float(bmi) * float(heart_rate)

def CardiovascularStressIndex(map ,heart_rate):
    return float(map) * float(heart_rate)
    