#impoting the libraries 
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pathlib 
import numpy as np
import pandas as pd
import os
from models_files.utils import *
#importing the models and scalers of it
BASE_DIR = pathlib.Path(__file__).resolve().parent / "saved model"

heart_model = joblib.load(BASE_DIR / "heart model.joblib")
heart_scaler = joblib.load(BASE_DIR / "heart_scaler.joblib")

kidney_model = joblib.load(BASE_DIR / "kidney model.joblib")
kidney_scaler = joblib.load(BASE_DIR / "kidney_scaler.joblib")

liver_model = joblib.load(BASE_DIR / "liver model.joblib")
liver_scaler = joblib.load(BASE_DIR / "liver_scaler.joblib")

stroke_model = joblib.load(BASE_DIR / "stroke model.joblib")
stroke_scaler = joblib.load(BASE_DIR / "stoke_scaler.joblib")

diabetes_model = joblib.load(BASE_DIR / "diabetes model.joblib")
diabetes_scaler = joblib.load(BASE_DIR / "diabetes_scaler.joblib")

anemia_model = joblib.load(BASE_DIR / "anemia model.joblib")
anemia_encoder = joblib.load(BASE_DIR / "anemia_encoder.joblib")
anemia_scaler = joblib.load(BASE_DIR / "anemia_scaler.joblib")

alzheimer_model = joblib.load(BASE_DIR / "alzheimer model.joblib")
alzheimer_scaler = joblib.load(BASE_DIR / "alzheimer_scaler.joblib")

chronic_model = joblib.load(BASE_DIR / "chronic model.joblib")
chronic_scaler = joblib.load(BASE_DIR / "chronic_scaler.joblib")

hyper_model = joblib.load(BASE_DIR / "hypertension model.joblib")
hyper_scaler = joblib.load(BASE_DIR / "hypertension_scaler.joblib")
#sarting the app settings with flask
FRONTEND_DIR = pathlib.Path(__file__).resolve().parent 
index_path =  FRONTEND_DIR / "index.html"
if index_path.exits():
    with open(index_path, 'r' ,encoding='utf-8') as file:
        html = file.read()
    print("success")
else:
    print(f"the file not in : {index_path}")
app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static"
)


@app.route("/") 
def index():
    return render_template(html)

@app.route("/api/predict/heart", methods=["POST"])
def heart_predictio.n():
    try:
        values = request.get_json(silent=True) or {}
        systolic_bp = float(values["systolic_bp"])
        diastolic_bp = float(values["diastolic_bp"])
        heart_rate = float(values["heart_rate"])
        cholesterol_mg_dl = float(values["cholesterol_mg_dl"])
        smoking_status = float(values["smoking"])
        alcohol_consumption = float(values["alcohol_consumption"])
        physical_activity = float(values["physical_activity"])
        bmi = float(values["bmi"])

        Map = MAP(systolic_bp, diastolic_bp)
        Rpp = RPP(systolic_bp, heart_rate)
        Pp = PP(systolic_bp, diastolic_bp)
        unhealthy_lifestyle_score = UnhealthyLifeScore(smoking_status, alcohol_consumption, physical_activity)
        atherogenic_index_coefficient = AtherogenicIndexCoefficient(cholesterol_mg_dl, systolic_bp)
        smoking_hypertension_interaction = SmokingHypertensionInteraction(smoking_status, systolic_bp)
        cardiac_adiposity_proxy = CardiacAdiposityProxy(bmi, heart_rate)
        cardiovascular_stress_index = CardiovascularStressIndex(Map, heart_rate)

        full_values = dict(values)
        full_values.update({
            "MAP": Map,
            "RPP Rate Pressure Product": Rpp,
            "PP Pulse Pressure": Pp,
            "unhealthy_lifestyle_score": unhealthy_lifestyle_score,
            "Atherogenic Index Coefficient": atherogenic_index_coefficient,
            "Smoking-Hypertension Interaction": smoking_hypertension_interaction,
            "Cardiac Adiposity Proxy": cardiac_adiposity_proxy,
            "Cardiovascular Stress Index": cardiovascular_stress_index,
        })

        cols = HEART_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in full_values:
                return jsonify({"error": f"Missing value for: {col}"}), 400
            ordered_row.append(float(full_values[col]))

        input_df = pd.DataFrame([ordered_row], columns=cols)
        scale_cols = [
            'age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp',
            'diastolic_bp', 'bmi', 'MAP',
            'RPP Rate Pressure Product', 'PP Pulse Pressure',
            'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction',
            'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index'
        ]
        unimportant_cols = ['gender', 'alcohol_consumption', 'heart_rate']
        final_input = scalingfortest(input_df, scale_cols, heart_scaler, unimportant_cols)
        proba = float(heart_model.predict(final_input, verbose=0)[0][0])
        risk_score = proba * 100
        confidence_pct = round((proba if proba >= 0.5 else (1.0 - proba)) * 100, 1)                
        message, recommendation = riskscore_messege(risk_score)
        level, level_text = leveltext_predict(risk_score)

        return jsonify({
            "riskScore": round(risk_score, 1),
            "level": level,
            "levelText": level_text,
            "detail": message,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": confidence_pct,
            "predictedLabel": level_text,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/kidney", methods=["POST"])
def kidney_prediction():
    try:
        values = request.get_json(silent=True) or {}
        values = dict(values)
        cols = KIDNEY_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in values:
                return jsonify({"error": f"Missing value for: {col}"}), 400
            ordered_row.append(values[col])

        input_df = pd.DataFrame([ordered_row], columns=cols)

        mapping_dict = {
            'Hypertension (yes/no)': {'yes': 1, 'no': 0, 1: 1, 0: 0},
            'Diabetes mellitus (yes/no)': {'yes': 1, 'no': 0, 1: 1, 0: 0},
            'Coronary artery disease (yes/no)': {'yes': 1, 'no': 0, 1: 1, 0: 0},
            'Pedal edema (yes/no)': {'yes': 1, 'no': 0, 1: 1, 0: 0},
            'Anemia (yes/no)': {'yes': 1, 'no': 0, 1: 1, 0: 0},
            'Family history of chronic kidney disease': {'yes': 1, 'no': 0, 1: 1, 0: 0},
            'Smoking status': {'yes': 1, 'no': 0, 1: 1, 0: 0},
            'Red blood cells in urine': {'abnormal': 1, 'normal': 0, 1: 1, 0: 0},
            'Pus cells in urine': {'abnormal': 1, 'normal': 0, 1: 1, 0: 0},
            'Urinary sediment microscopy results': {'abnormal': 1, 'normal': 0, 1: 1, 0: 0},
            'Pus cell clumps in urine': {'present': 1, 'not present': 0, 1: 1, 0: 0},
            'Bacteria in urine': {'present': 1, 'not present': 0, 1: 1, 0: 0},
            'Appetite (good/poor)': {'poor': 1, 'good': 0, 1: 1, 0: 0},
            'Physical activity level': {'high': 2, 'moderate': 1, 'low': 0, 2: 2, 1: 1, 0: 0}
        }

        for col_name, map_vals in mapping_dict.items():
            if col_name in input_df.columns:
                input_df[col_name] = input_df[col_name].map(lambda x: map_vals.get(x, x))

        scaled_cols = [
            'Age of the patient', 'Blood pressure (mm/Hg)', 'Specific gravity of urine',
            'Albumin in urine', 'Sugar in urine', 'Random blood glucose level (mg/dl)',
            'Blood urea (mg/dl)', 'Serum creatinine (mg/dl)', 'Sodium level (mEq/L)',
            'Potassium level (mEq/L)', 'Hemoglobin level (gms)', 'Packed cell volume (%)',
            'White blood cell count (cells/cumm)', 'Red blood cell count (millions/cumm)',
            'Estimated Glomerular Filtration Rate (eGFR)', 'Urine protein-to-creatinine ratio',
            'Urine output (ml/day)', 'Serum albumin level', 'Cholesterol level',
            'Parathyroid hormone (PTH) level', 'Serum calcium level', 'Serum phosphate level',
            'Body Mass Index (BMI)', 'Physical activity level',
            'Duration of diabetes mellitus (years)', 'Duration of hypertension (years)',
            'Cystatin C level', 'C-reactive protein (CRP) level', 'Interleukin-6 (IL-6) level'
        ]

        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        final_input = scalingfortest(input_df, scaled_cols, kidney_scaler, None)
        probabilities = kidney_model.predict(final_input, verbose=0)[0]
        
        predicted_index = int(np.argmax(probabilities))
        confidence_pct = round(float(probabilities[predicted_index]) * 100, 1)

        if predicted_index == 2:
            level = "high"
            level_text = "High Risk"
            message = "The indicators suggest a high risk of chronic kidney disease."
            recommendation = "Medical consultation is recommended as soon as possible."
            risk_score = round(float(probabilities[2]) * 100, 1)
            detail = "This is percentage for if it is the patient with high risk"
        elif predicted_index == 1:
            level = "moderate"
            level_text = "Moderate Risk"
            message = "The indicators suggest a moderate risk of chronic kidney disease."
            recommendation = "Close medical follow-up and a repeat test soon are recommended."
            risk_score = round(float(probabilities[1]) * 100, 1)
            detail = "This is percentage for if it is the patient with moderate risk"
        else:
            level = "low"
            level_text = "Low Risk"
            message = "The indicators suggest a low risk."
            recommendation = "Continue with routine monitoring."
            risk_score = round(float(probabilities[0]) * 100, 1)
            detail = "This is percentage for if it is the patient with low risk"

        return jsonify({
            "riskScore": risk_score,
            "level": level,
            "levelText": level_text,
            "detail": detail,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": confidence_pct,
            "predictedLabel": level_text,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/anemia", methods=["POST"])
def anemia_prediction():
    try:
        values = request.get_json(silent=True) or {}
        values = dict(values)
        cols = ANEMIA_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in values:
                return jsonify({"error": f"Missing value for: {col}"}), 400
            ordered_row.append(float(values[col]))
        input_df = pd.DataFrame([ordered_row], columns=cols)
        scaled_cols = ['WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT']
        final_input = scalingfortest(input_df, scaled_cols, anemia_scaler, None)
        probabilities = anemia_model.predict(final_input, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence_pct = round(float(probabilities[predicted_index]) * 100, 1)
        risk_score = round(float(probabilities[predicted_index]) * 100, 1)
        output = str(anemia_encoder.inverse_transform([predicted_index])[0])
        if output == "Healthy":
            level = "Low"
            level_text = "Low Risk"
            message = f"The indicators suggest a Low risk of anemia and the patient state is : {output}"
            recommendation = "Continue with routine monitoring."
            detail = ""
        elif output == "Normocytic hypochromic anemia":
            level = "High"
            level_text = "High Risk"
            message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
            _,recommendation = riskscore_messege(risk_score)
            detail = ""
        elif output == "Iron deficiency anemia":
            level = "High"
            level_text = "High Risk"
            message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
            _,recommendation = riskscore_messege(risk_score)
            detail = ""
        elif output == "Other microcytic anemia":
            level = "High"
            level_text = "High Risk"
            message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
            _,recommendation = riskscore_messege(risk_score)
            detail = ""
        elif output == "Leukemia":
            level = "High"
            level_text = "High Risk"
            message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
            _,recommendation = riskscore_messege(risk_score)
            detail = ""
        elif output == "Thrombocytopenia":
            level = "High"
            level_text = "High Risk"
            message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
            _,recommendation = riskscore_messege(risk_score)
            detail = ""
        elif output == "Normocytic normochromic anemia":
            level = "High"
            level_text = "High Risk"
            message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
            _,recommendation = riskscore_messege(risk_score)
            detail = ""
        elif output == "Macrocytic anemia":
            level = "High"
            level_text = "High Risk"
            message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
            _,recommendation = riskscore_messege(risk_score)
            detail = ""
        elif output == "Leukemia with thrombocytopenia":
            level = "High"
            level_text = "High Risk"
            message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
            _,recommendation = riskscore_messege(risk_score)
            detail = ""
        return jsonify({
            "riskScore": risk_score,
            "level": level,
            "levelText": level_text,
            "detail": detail,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": float(confidence_pct),
            "predictedLabel": output,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/stroke" ,methods=["POST"])
def stroke_prediction():
    try:
        values = request.get_json(silent=True) or {}
        values = dict(values)
        cols = STROKE_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in values:
                return jsonify({"error": f"Missing value for: {col}"}), 400
            ordered_row.append(values[col])
        input_df = pd.DataFrame([ordered_row], columns=cols)
        input_df['gender'] = replace_values_in_csv(input_df ,'gender' ,"Female" ,'Male' ,"Other")
        input_df['ever_married'] = replace_values_in_csv(input_df ,'ever_married' ,"No" ,"Yes" ,None)
        input_df['Residence_type'] = replace_values_in_csv(input_df ,'Residence_type' ,"Rural" ,"Urban" ,None)  
        smoking_map = {
            "never smoked": 0,
            "formerly smoked": 1,
            "Unknown": 1,
            "smokes": 2
        }
        input_df['smoking_status'] = input_df['smoking_status'].map(smoking_map).astype(int)
        work_type_map = {
            "Private": 0,
            "Self-employed": 2,
            "Govt_job": 2,
            "children": 3,
            "Never_worked": 4
        }
        input_df['work_type'] = input_df['work_type'].map(work_type_map).astype(int)
        scaled_cols = ['age', 'work_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        final_input = scalingfortest(input_df, scaled_cols, stroke_scaler, None)
        proba = float(stroke_model.predict(final_input, verbose=0)[0][0])
        risk_score = round(proba * 100, 1)
        confidence_pct = round((proba if proba >= 0.5 else (1 - proba)) * 100, 1)
        message, recommendation = riskscore_messege(risk_score)
        level, level_text = leveltext_predict(risk_score)
        return jsonify({
            "riskScore": risk_score,
            "level": level,
            "levelText": level_text,
            "detail": message,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": confidence_pct,
            "predictedLabel": risk_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/predict/diabetes" ,methods=["Post"])
def diabetes_prediction():
    try:
        values = request.get_json(silent=True) or {}
        values = dict(values)
        cols = DIABETES_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in values:
                return jsonify({"error" : f"Missinf value for {col}"}), 400
            ordered_row.append(values[col])
        input_df = pd.DataFrame([ordered_row], columns=cols)
        input_df['gender'] = replace_values_in_csv(input_df ,'gender' ,"Female" ,"Male" ,"Other")
        smoking_values ={
            "never":0,
            "not current":1,
            "current":3,
            "former":2,
            "ever":1,
            "No Info":1
        }
        input_df['smoking_history'] = input_df['smoking_history'].replace(smoking_values).astype(int)
        for col in cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        scaled_cols = ['age' ,'smoking_history' ,'bmi' ,'HbA1c_level' ,'blood_glucose_level']
        final_input = scalingfortest(input_df ,scaled_cols ,diabetes_scaler ,None)
        proba = float(diabetes_model.predict(final_input, verbose=0)[0][0])
        risk_score = round(proba * 100)
        confidence_pct = round((proba if proba >= 0.5 else (1 - proba) ) * 100, 1)
        message ,recommendation = riskscore_messege(risk_score)
        level ,level_text = leveltext_predict(risk_score)
        return jsonify({
            "riskScore": risk_score,
            "level": level,
            "levelText": level_text,
            "detail": message,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": confidence_pct,
            "predictedLabel": risk_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/predict/liver" ,methods=["Post"])
def liver_prediction():
    try:
        values = request.get_json(silent=True) or {}
        values = dict(values)
        cols = LIVER_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in values:
                return jsonify({"error": f"Missing value for {col}"}), 400
            ordered_row.append(values[col])
        input_df = pd.DataFrame([ordered_row], columns=cols)
        input_df['gender'] = replace_values_in_csv(input_df ,'gender' ,"Female" ,"Male" ,None)
        scaled_cols = ['age' ,'tot_bilirubin' ,'direct_bilirubin' ,'tot_proteins' ,'albumin' ,'ag_ratio' ,'sgpt' ,'sgot' ,'alkphos']
        final_input = scalingfortest(input_df ,scaled_cols ,liver_scaler ,None)
        proba = float(liver_model.predict(final_input ,verbose=0)[0][0])
        risk_score = proba * 100
        confidence_pct = round((proba if proba >=0.5 else (1 - proba)) * 100 ,1)
        message ,recommendation = riskscore_messege(risk_score)
        level ,level_text = leveltext_predict(risk_score)
        return jsonify({
            "riskScore": risk_score,
            "level": level,
            "levelText": level_text,
            "detail": message,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": confidence_pct,
            "predictedLabel": risk_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/alzheimer" ,methods=["Post"])
def alzheimer_prediction():
    try:
        values = request.get_json(silent=True) or {}
        values = dict(values)
        cols = ALZHEIMER_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in values:
                return jsonify({"error": f"Missing value : {col}"}),400
            ordered_row.append(values[col])
        input_df = pd.DataFrame([ordered_row], columns=cols)
        input_df = input_df.apply(pd.to_numeric, errors="coerce")
        input_df = input_df.astype(np.float32)
        scaled_cols = ['Age' ,'Ethnicity' ,'EducationLevel' ,'BMI','AlcoholConsumption' ,'PhysicalActivity' ,'DietQuality' ,'SleepQuality' ,'SystolicBP' ,'DiastolicBP' ,'CholesterolTotal' ,'CholesterolLDL' ,'CholesterolHDL' ,'CholesterolTriglycerides' ,'MMSE' ,'FunctionalAssessment' ,'ADL']
        final_input = scalingfortest(input_df ,scaled_cols ,alzheimer_scaler ,None)
        proba = float(alzheimer_model.predict(final_input ,verbose=0)[0][0])
        risk_score = proba * 100
        confidence_pct = round((proba if proba >=0.5 else (1 - proba)) * 100 ,1)
        message ,recommendation = riskscore_messege(risk_score)
        level ,level_text = leveltext_predict(risk_score) 
        return jsonify({
            "riskScore": risk_score,
            "level": level,
            "levelText": level_text,
            "detail": message,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": confidence_pct,
            "predictedLabel": risk_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/predict/chronic_kidney" ,methods=["Post"])
def chronic_kidney():
    try:
        values = request.get_json(silent=True) or {}
        values = dict(values)
        cols = CHRONIC_KIDNEY_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in values:
                return jsonify({"error": f"Missing value : {col}"}),400
            ordered_row.append(values[col])
        input_df = pd.DataFrame([ordered_row], columns=cols)
        input_df = input_df.apply(pd.to_numeric, errors="coerce")
        input_df = input_df.astype(np.float32)
        scaled_cols = ['Bp' ,'Sg' ,'Al' ,'Su' ,'Pot', 'Bu' ,'Sc' ,'Sod' ,'Hemo' ,'Wbcc' ,'Rbcc']
        final_input = scalingfortest(input_df ,scaled_cols ,chronic_scaler ,None)
        proba = float(chronic_model.predict(final_input ,verbose=0)[0][0])
        risk_score = proba * 100
        confidence_pct = round((proba if proba >=0.5 else (1 - proba)) * 100 ,1)
        message ,recommendation = riskscore_messege(risk_score)
        level ,level_text = leveltext_predict(risk_score)
        return jsonify({
            "riskScore": risk_score,
            "level": level,
            "levelText": level_text,
            "detail": message,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": confidence_pct,
            "predictedLabel": risk_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/hypertension" ,methods=["Post"])
def hypertension():
    try:
        values = request.get_json(silent=True) or {}
        values = dict(values)
        cols = HYPERTENSION_COLUMNS
        ordered_row = []
        for col in cols:
            if col not in values:
                return jsonify({"error": f"Missing value : {col}"}),400
            ordered_row.append(values[col])
        input_df = pd.DataFrame([ordered_row], columns=cols)
        medication_map = {
            "ACE Inhibitor": 0,
            "Beta Blocker": 1,
            "Diuretic": 2,
            "Other": 3,
            "None": 4,    
        }
        input_df['BP_History'] = replace_values_in_csv(input_df ,'BP_History' ,"Normal" ,"Prehypertension" ,"Hypertension")
        input_df['Exercise_Level'] = replace_values_in_csv(input_df ,'Exercise_Level' ,"Low" ,"Moderate" ,"High")
        input_df['Family_History'] = replace_values_in_csv(input_df ,'Family_History' ,"NO" ,"Yes" ,None)
        input_df['Smoking_Status'] = replace_values_in_csv(input_df ,'Smoking_Status' ,"Non-Smoker" ,"Smoker" ,None)
        input_df['Medication'] = input_df['Medication'].map(medication_map).astype(int)
        scaled_cols = ['Age' ,'Salt_Intake' ,'Stress_Score' ,'BP_History' ,'Sleep_Duration' ,'BMI' ,'Medication' ,'Exercise_Level']
        final_input = scalingfortest(input_df ,scaled_cols ,hyper_scaler ,None)
        proba = float(hyper_model.predict(final_input ,verbose=0)[0][0])
        risk_score = proba * 100
        confidence_pct = round((proba if proba >=0.5 else (1 - proba)) * 100 ,1)
        message ,recommendation = riskscore_messege(risk_score)
        level ,level_text = leveltext_predict(risk_score)
        return jsonify({
            "riskScore": risk_score,
            "level": level,
            "levelText": level_text,
            "detail": message,
            "message": message,
            "recommendation": recommendation,
            "confidencePct": confidence_pct,
            "predictedLabel": risk_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

'''
for start the web to work the :

cd backend
$env:FLASK_DEBUG = "1"
   flask run
''' 
