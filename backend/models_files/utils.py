#importing initial libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import pathlib 
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Defining the functions of the data preprocessing an scaling
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

#Defining the function of calculating the advanced features of heart attack model
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

def riskscore_messege(risk_score):
    message = None
    recommendation = None
    if risk_score >= 80:
        message = "The indicators suggest a very high risk."
        recommendation = "Please consult a specialist immediately and get confirmatory tests."
    elif risk_score >= 66:
        message = "The indicators suggest a high risk."
        recommendation = "The Medical Consultation is recommended as soon as possible."
    elif risk_score >= 50:
        message = "The indicators suggest a moderate-to-high risk."
        recommendation = "Close medical follow-up and a repeat test soon are recommended."
    elif risk_score >= 33:
        message = "The indicators suggest a moderate risk."
        recommendation = "Routine follow-up and lifestyle adjustments are recommended."
    elif risk_score >= 15:
        message = "The indicators suggest a low risk."
        recommendation = "Continue with routine monitoring."
    else:
        message = "The values fall within the normal range."
        recommendation = "No further action needed at this time; annual routine checkup is sufficient."
    return message ,recommendation

def leveltext_predict(risk_score):
    if risk_score >= 66:
        level, level_text = "high", "High Risk"
    elif risk_score >= 30:
        level, level_text = "moderate", "Moderate Risk"
    elif risk_score >= 15:
        level, level_text = "low", "Low Risk"
    else:
        level, level_text = "Normal", "Normal precentage"
    return level ,level_text
#write the columns of each model input
HEART_COLUMNS = [
    "age", "gender", "glucose_mg_dl", "cholesterol_mg_dl", "systolic_bp",
    "diastolic_bp", "heart_rate", "alcohol_consumption", "smoking",
    "bmi", "physical_activity", "family_history", "MAP",
    "RPP Rate Pressure Product", "PP Pulse Pressure", "unhealthy_lifestyle_score",
    "Atherogenic Index Coefficient", "Smoking-Hypertension Interaction",
    "Cardiac Adiposity Proxy", "Cardiovascular Stress Index"
]
ANEMIA_COLUMNS = [
    "WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT",
    "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT"
]
STROKE_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
]
KIDNEY_COLUMNS = [
    "Age of the patient", "Blood pressure (mm/Hg)", "Specific gravity of urine",
    "Albumin in urine", "Sugar in urine", "Red blood cells in urine",
    "Pus cells in urine", "Pus cell clumps in urine", "Bacteria in urine",
    "Random blood glucose level (mg/dl)", "Blood urea (mg/dl)",
    "Serum creatinine (mg/dl)", "Sodium level (mEq/L)", "Potassium level (mEq/L)",
    "Hemoglobin level (gms)", "Packed cell volume (%)",
    "White blood cell count (cells/cumm)", "Red blood cell count (millions/cumm)",
    "Hypertension (yes/no)", "Diabetes mellitus (yes/no)",
    "Coronary artery disease (yes/no)", "Appetite (good/poor)",
    "Pedal edema (yes/no)", "Anemia (yes/no)",
    "Estimated Glomerular Filtration Rate (eGFR)",
    "Urine protein-to-creatinine ratio", "Urine output (ml/day)",
    "Serum albumin level", "Cholesterol level",
    "Parathyroid hormone (PTH) level", "Serum calcium level",
    "Serum phosphate level", "Family history of chronic kidney disease",
    "Smoking status", "Body Mass Index (BMI)", "Physical activity level",
    "Duration of diabetes mellitus (years)", "Duration of hypertension (years)",
    "Cystatin C level", "Urinary sediment microscopy results",
    "C-reactive protein (CRP) level", "Interleukin-6 (IL-6) level"
]
DIABETES_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level"
]
LIVER_COLUMNS = [
    "age", "gender", "tot_bilirubin", "direct_bilirubin", "tot_proteins",
    "albumin", "ag_ratio", "sgpt", "sgot", "alkphos"
]
reverse_risk_mapping = {  # that for kidney output
    0: "Low Risk (No Disease / Low Risk)",
    1: "Moderate Risk",
    2: "High Risk (High Risk / Severe Disease)",
}
ALZHEIMER_COLUMNS = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
    'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
    'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
    'Confusion', 'Disorientation', 'PersonalityChanges',
    'DifficultyCompletingTasks', 'Forgetfulness'
]

CHRONIC_KIDNEY_COLUMNS = [
    'Bp', 'Sg', 'Al', 'Su', 'Rbc', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Wbcc', 'Rbcc', 'Htn'
]

HYPERTENSION_COLUMNS = [
    'Age', 'Salt_Intake', 'Stress_Score', 'BP_History', 'Sleep_Duration', 'BMI',
    'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status'
]
    