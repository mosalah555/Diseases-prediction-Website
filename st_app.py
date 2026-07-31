# importing the libraries
import pathlib
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from models_files.utils import *
 
# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Disease Risk Predictor",
    page_icon="🩺",
    layout="centered",
)
 
# ---------------------------------------------------------------------------
# Loading the models and scalers (cached so they only load once per session)
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent / "backend" / "saved model"
 
 
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {}
 
    models["heart_model"] = joblib.load(BASE_DIR / "heart model.joblib")
    models["heart_scaler"] = joblib.load(BASE_DIR / "heart_scaler.joblib")
 
    models["kidney_model"] = joblib.load(BASE_DIR / "kidney model.joblib")
    models["kidney_scaler"] = joblib.load(BASE_DIR / "kidney_scaler.joblib")
 
    models["liver_model"] = joblib.load(BASE_DIR / "liver model.joblib")
    models["liver_scaler"] = joblib.load(BASE_DIR / "liver_scaler.joblib")
 
    models["stroke_model"] = joblib.load(BASE_DIR / "stroke model.joblib")
    models["stroke_scaler"] = joblib.load(BASE_DIR / "stoke_scaler.joblib")
 
    models["diabetes_model"] = joblib.load(BASE_DIR / "diabetes model.joblib")
    models["diabetes_scaler"] = joblib.load(BASE_DIR / "diabetes_scaler.joblib")
 
    models["anemia_model"] = joblib.load(BASE_DIR / "anemia model.joblib")
    models["anemia_encoder"] = joblib.load(BASE_DIR / "anemia_encoder.joblib")
    models["anemia_scaler"] = joblib.load(BASE_DIR / "anemia_scaler.joblib")
 
    models["alzheimer_model"] = joblib.load(BASE_DIR / "alzheimer model.joblib")
    models["alzheimer_scaler"] = joblib.load(BASE_DIR / "alzheimer_scaler.joblib")
 
    models["chronic_model"] = joblib.load(BASE_DIR / "chronic model.joblib")
    models["chronic_scaler"] = joblib.load(BASE_DIR / "chronic_scaler.joblib")
 
    models["hyper_model"] = joblib.load(BASE_DIR / "hypertension model.joblib")
    models["hyper_scaler"] = joblib.load(BASE_DIR / "hypertension_scaler.joblib")
 
    return models
 
 
M = load_models()
 
# ---------------------------------------------------------------------------
# Small helper to render the result card the same way for every predictor
# ---------------------------------------------------------------------------
def render_result(risk_score, level, level_text, message, recommendation, confidence_pct, predicted_label=None):
    level_l = str(level).lower()
    if level_l in ("high",):
        color = "#e74c3c"
    elif level_l in ("moderate", "medium"):
        color = "#f39c12"
    else:
        color = "#27ae60"
 
    st.markdown(
        f"""
        <div style="border-left:6px solid {color};padding:16px 20px;border-radius:10px;
                    background-color:rgba(127,127,127,0.08);margin-top:12px;">
            <h3 style="margin:0 0 6px 0;color:{color};">{level_text}</h3>
            <p style="margin:0;font-size:15px;"><b>Risk score:</b> {risk_score}</p>
            <p style="margin:4px 0 0 0;font-size:15px;"><b>Confidence:</b> {confidence_pct}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(f"**Detail:** {message}")
    st.write(f"**Recommendation:** {recommendation}")
    if predicted_label is not None:
        st.caption(f"Predicted label: {predicted_label}")
 
 
def build_ordered_row(values: dict, cols):
    """Same 'missing column' guard as the Flask version, but raises for Streamlit to catch."""
    ordered_row = []
    for col in cols:
        if col not in values:
            raise KeyError(f"Missing value for: {col}")
        ordered_row.append(values[col])
    return ordered_row
 
 
# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 Disease Risk Predictor")
page = st.sidebar.radio(
    "Choose a predictor",
    [
        "Heart",
        "Kidney",
        "Anemia",
        "Stroke",
        "Diabetes",
        "Liver",
        "Alzheimer",
        "Chronic Kidney",
        "Hypertension",
    ],
)
 
st.title(f"{page} Risk Prediction")
 
# ---------------------------------------------------------------------------
# HEART
# ---------------------------------------------------------------------------
if page == "Heart":
    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 120, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            systolic_bp = st.number_input("Systolic BP", 60.0, 260.0, 120.0)
            diastolic_bp = st.number_input("Diastolic BP", 40.0, 180.0, 80.0)
            heart_rate = st.number_input("Heart rate", 30.0, 220.0, 75.0)
            glucose_mg_dl = st.number_input("Glucose (mg/dl)", 40.0, 500.0, 100.0)
        with col2:
            cholesterol_mg_dl = st.number_input("Cholesterol (mg/dl)", 80.0, 500.0, 190.0)
            smoking = st.selectbox("Smoking status", [0, 1], format_func=lambda x: "Smoker" if x else "Non-smoker")
            alcohol_consumption = st.number_input("Alcohol consumption (units/week)", 0.0, 50.0, 2.0)
            physical_activity = st.number_input("Physical activity (hrs/week)", 0.0, 40.0, 3.0)
            bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
 
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            values = {
                "age": age,
                "gender": 1 if gender == "Male" else 0,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "heart_rate": heart_rate,
                "glucose_mg_dl": glucose_mg_dl,
                "cholesterol_mg_dl": cholesterol_mg_dl,
                "smoking": smoking,
                "alcohol_consumption": alcohol_consumption,
                "physical_activity": physical_activity,
                "bmi": bmi,
            }
 
            Map = MAP(systolic_bp, diastolic_bp)
            Rpp = RPP(systolic_bp, heart_rate)
            Pp = PP(systolic_bp, diastolic_bp)
            unhealthy_lifestyle_score = UnhealthyLifeScore(smoking, alcohol_consumption, physical_activity)
            atherogenic_index_coefficient = AtherogenicIndexCoefficient(cholesterol_mg_dl, systolic_bp)
            smoking_hypertension_interaction = SmokingHypertensionInteraction(smoking, systolic_bp)
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
                    st.error(f"Missing value for: {col}")
                    st.stop()
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
            final_input = scalingfortest(input_df, scale_cols, M["heart_scaler"], unimportant_cols)
            proba = float(M["heart_model"].predict(final_input, verbose=0)[0][0])
            risk_score = proba * 100
            confidence_pct = round((proba if proba >= 0.5 else (1.0 - proba)) * 100, 1)
            message, recommendation = riskscore_messege(risk_score)
            level, level_text = leveltext_predict(risk_score)
 
            render_result(round(risk_score, 1), level, level_text, message, recommendation, confidence_pct, level_text)
        except Exception as e:
            st.error(f"Error: {e}")
 
# ---------------------------------------------------------------------------
# KIDNEY
# ---------------------------------------------------------------------------
elif page == "Kidney":
    st.info("Fill in the patient values below (categorical fields use dropdowns).")
    with st.form("kidney_form"):
        cols_meta = KIDNEY_COLUMNS
        yes_no_cols = [
            'Hypertension (yes/no)', 'Diabetes mellitus (yes/no)', 'Coronary artery disease (yes/no)',
            'Pedal edema (yes/no)', 'Anemia (yes/no)', 'Family history of chronic kidney disease',
            'Smoking status'
        ]
        abnormal_normal_cols = ['Red blood cells in urine', 'Pus cells in urine', 'Urinary sediment microscopy results']
        present_cols = ['Pus cell clumps in urine', 'Bacteria in urine']
        appetite_col = 'Appetite (good/poor)'
        activity_col = 'Physical activity level'
 
        values = {}
        for col in cols_meta:
            if col in yes_no_cols:
                values[col] = st.selectbox(col, ["no", "yes"])
            elif col in abnormal_normal_cols:
                values[col] = st.selectbox(col, ["normal", "abnormal"])
            elif col in present_cols:
                values[col] = st.selectbox(col, ["not present", "present"])
            elif col == appetite_col:
                values[col] = st.selectbox(col, ["good", "poor"])
            elif col == activity_col:
                values[col] = st.selectbox(col, ["low", "moderate", "high"])
            else:
                values[col] = st.number_input(col, value=0.0)
 
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            cols = KIDNEY_COLUMNS
            ordered_row = build_ordered_row(values, cols)
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
 
            final_input = scalingfortest(input_df, scaled_cols, M["kidney_scaler"], None)
            probabilities = M["kidney_model"].predict(final_input, verbose=0)[0]
 
            predicted_index = int(np.argmax(probabilities))
            confidence_pct = round(float(probabilities[predicted_index]) * 100, 1)
 
            if predicted_index == 2:
                level, level_text = "high", "High Risk"
                message = "The indicators suggest a high risk of chronic kidney disease."
                recommendation = "Medical consultation is recommended as soon as possible."
                risk_score = round(float(probabilities[2]) * 100, 1)
            elif predicted_index == 1:
                level, level_text = "moderate", "Moderate Risk"
                message = "The indicators suggest a moderate risk of chronic kidney disease."
                recommendation = "Close medical follow-up and a repeat test soon are recommended."
                risk_score = round(float(probabilities[1]) * 100, 1)
            else:
                level, level_text = "low", "Low Risk"
                message = "The indicators suggest a low risk."
                recommendation = "Continue with routine monitoring."
                risk_score = round(float(probabilities[0]) * 100, 1)
 
            render_result(risk_score, level, level_text, message, recommendation, confidence_pct, level_text)
        except Exception as e:
            st.error(f"Error: {e}")
 
# ---------------------------------------------------------------------------
# ANEMIA
# ---------------------------------------------------------------------------
elif page == "Anemia":
    with st.form("anemia_form"):
        cols = ANEMIA_COLUMNS
        values = {}
        for col in cols:
            values[col] = st.number_input(col, value=0.0)
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            cols = ANEMIA_COLUMNS
            ordered_row = [float(values[c]) for c in cols]
            input_df = pd.DataFrame([ordered_row], columns=cols)
            scaled_cols = ['WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT']
            final_input = scalingfortest(input_df, scaled_cols, M["anemia_scaler"], None)
            probabilities = M["anemia_model"].predict(final_input, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            confidence_pct = round(float(probabilities[predicted_index]) * 100, 1)
            risk_score = round(float(probabilities[predicted_index]) * 100, 1)
            output = str(M["anemia_encoder"].inverse_transform([predicted_index])[0])
 
            if output == "Healthy":
                level, level_text = "Low", "Low Risk"
                message = f"The indicators suggest a Low risk of anemia and the patient state is : {output}"
                recommendation = "Continue with routine monitoring."
            else:
                level, level_text = "High", "High Risk"
                message = f"The indicators suggest a high risk of anemia and the type of it : {output}"
                _, recommendation = riskscore_messege(risk_score)
 
            render_result(risk_score, level, level_text, message, recommendation, confidence_pct, output)
        except Exception as e:
            st.error(f"Error: {e}")
 
# ---------------------------------------------------------------------------
# STROKE
# ---------------------------------------------------------------------------
elif page == "Stroke":
    with st.form("stroke_form"):
        cols_meta = STROKE_COLUMNS
        values = {}
        for col in cols_meta:
            if col == "gender":
                values[col] = st.selectbox("Gender", ["Male", "Female", "Other"])
            elif col == "ever_married":
                values[col] = st.selectbox("Ever married", ["Yes", "No"])
            elif col == "Residence_type":
                values[col] = st.selectbox("Residence type", ["Urban", "Rural"])
            elif col == "smoking_status":
                values[col] = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
            elif col == "work_type":
                values[col] = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            else:
                values[col] = st.number_input(col, value=0.0)
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            cols = STROKE_COLUMNS
            ordered_row = build_ordered_row(values, cols)
            input_df = pd.DataFrame([ordered_row], columns=cols)
            input_df['gender'] = replace_values_in_csv(input_df, 'gender', "Female", 'Male', "Other")
            input_df['ever_married'] = replace_values_in_csv(input_df, 'ever_married', "No", "Yes", None)
            input_df['Residence_type'] = replace_values_in_csv(input_df, 'Residence_type', "Rural", "Urban", None)
            smoking_map = {"never smoked": 0, "formerly smoked": 1, "Unknown": 1, "smokes": 2}
            input_df['smoking_status'] = input_df['smoking_status'].map(smoking_map).astype(int)
            work_type_map = {"Private": 0, "Self-employed": 2, "Govt_job": 2, "children": 3, "Never_worked": 4}
            input_df['work_type'] = input_df['work_type'].map(work_type_map).astype(int)
            scaled_cols = ['age', 'work_type', 'avg_glucose_level', 'bmi', 'smoking_status']
            final_input = scalingfortest(input_df, scaled_cols, M["stroke_scaler"], None)
            proba = float(M["stroke_model"].predict(final_input, verbose=0)[0][0])
            risk_score = round(proba * 100, 1)
            confidence_pct = round((proba if proba >= 0.5 else (1 - proba)) * 100, 1)
            message, recommendation = riskscore_messege(risk_score)
            level, level_text = leveltext_predict(risk_score)
            render_result(risk_score, level, level_text, message, recommendation, confidence_pct, risk_score)
        except Exception as e:
            st.error(f"Error: {e}")
 
# ---------------------------------------------------------------------------
# DIABETES
# ---------------------------------------------------------------------------
elif page == "Diabetes":
    with st.form("diabetes_form"):
        cols_meta = DIABETES_COLUMNS
        values = {}
        for col in cols_meta:
            if col == "gender":
                values[col] = st.selectbox("Gender", ["Male", "Female", "Other"])
            elif col == "smoking_history":
                values[col] = st.selectbox("Smoking history", ["never", "not current", "current", "former", "ever", "No Info"])
            else:
                values[col] = st.number_input(col, value=0.0)
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            cols = DIABETES_COLUMNS
            ordered_row = build_ordered_row(values, cols)
            input_df = pd.DataFrame([ordered_row], columns=cols)
            input_df['gender'] = replace_values_in_csv(input_df, 'gender', "Female", "Male", "Other")
            smoking_values = {"never": 0, "not current": 1, "current": 3, "former": 2, "ever": 1, "No Info": 1}
            input_df['smoking_history'] = input_df['smoking_history'].replace(smoking_values).astype(int)
            for col in cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            scaled_cols = ['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            final_input = scalingfortest(input_df, scaled_cols, M["diabetes_scaler"], None)
            proba = float(M["diabetes_model"].predict(final_input, verbose=0)[0][0])
            risk_score = round(proba * 100)
            confidence_pct = round((proba if proba >= 0.5 else (1 - proba)) * 100, 1)
            message, recommendation = riskscore_messege(risk_score)
            level, level_text = leveltext_predict(risk_score)
            render_result(risk_score, level, level_text, message, recommendation, confidence_pct, risk_score)
        except Exception as e:
            st.error(f"Error: {e}")
 
# ---------------------------------------------------------------------------
# LIVER
# ---------------------------------------------------------------------------
elif page == "Liver":
    with st.form("liver_form"):
        cols_meta = LIVER_COLUMNS
        values = {}
        for col in cols_meta:
            if col == "gender":
                values[col] = st.selectbox("Gender", ["Male", "Female"])
            else:
                values[col] = st.number_input(col, value=0.0)
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            cols = LIVER_COLUMNS
            ordered_row = build_ordered_row(values, cols)
            input_df = pd.DataFrame([ordered_row], columns=cols)
            input_df['gender'] = replace_values_in_csv(input_df, 'gender', "Female", "Male", None)
            scaled_cols = ['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']
            final_input = scalingfortest(input_df, scaled_cols, M["liver_scaler"], None)
            proba = float(M["liver_model"].predict(final_input, verbose=0)[0][0])
            risk_score = proba * 100
            confidence_pct = round((proba if proba >= 0.5 else (1 - proba)) * 100, 1)
            message, recommendation = riskscore_messege(risk_score)
            level, level_text = leveltext_predict(risk_score)
            render_result(round(risk_score, 1), level, level_text, message, recommendation, confidence_pct, risk_score)
        except Exception as e:
            st.error(f"Error: {e}")
 
# ---------------------------------------------------------------------------
# ALZHEIMER
# ---------------------------------------------------------------------------
elif page == "Alzheimer":
    with st.form("alzheimer_form"):
        cols = ALZHEIMER_COLUMNS
        values = {}
        for col in cols:
            values[col] = st.number_input(col, value=0.0)
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            cols = ALZHEIMER_COLUMNS
            ordered_row = build_ordered_row(values, cols)
            input_df = pd.DataFrame([ordered_row], columns=cols)
            input_df = input_df.apply(pd.to_numeric, errors="coerce")
            input_df = input_df.astype(np.float32)
            scaled_cols = ['Age', 'Ethnicity', 'EducationLevel', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                           'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
                           'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
                           'FunctionalAssessment', 'ADL']
            final_input = scalingfortest(input_df, scaled_cols, M["alzheimer_scaler"], None)
            proba = float(M["alzheimer_model"].predict(final_input, verbose=0)[0][0])
            risk_score = proba * 100
            confidence_pct = round((proba if proba >= 0.5 else (1 - proba)) * 100, 1)
            message, recommendation = riskscore_messege(risk_score)
            level, level_text = leveltext_predict(risk_score)
            render_result(round(risk_score, 1), level, level_text, message, recommendation, confidence_pct, risk_score)
        except Exception as e:
            st.error(f"Error: {e}")
 
# ---------------------------------------------------------------------------
# CHRONIC KIDNEY
# ---------------------------------------------------------------------------
elif page == "Chronic Kidney":
    with st.form("chronic_kidney_form"):
        cols = CHRONIC_KIDNEY_COLUMNS
        values = {}
        for col in cols:
            values[col] = st.number_input(col, value=0.0)
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            cols = CHRONIC_KIDNEY_COLUMNS
            ordered_row = build_ordered_row(values, cols)
            input_df = pd.DataFrame([ordered_row], columns=cols)
            input_df = input_df.apply(pd.to_numeric, errors="coerce")
            input_df = input_df.astype(np.float32)
            scaled_cols = ['Bp', 'Sg', 'Al', 'Su', 'Pot', 'Bu', 'Sc', 'Sod', 'Hemo', 'Wbcc', 'Rbcc']
            final_input = scalingfortest(input_df, scaled_cols, M["chronic_scaler"], None)
            proba = float(M["chronic_model"].predict(final_input, verbose=0)[0][0])
            risk_score = proba * 100
            confidence_pct = round((proba if proba >= 0.5 else (1 - proba)) * 100, 1)
            message, recommendation = riskscore_messege(risk_score)
            level, level_text = leveltext_predict(risk_score)
            render_result(round(risk_score, 1), level, level_text, message, recommendation, confidence_pct, risk_score)
        except Exception as e:
            st.error(f"Error: {e}")
 
# ---------------------------------------------------------------------------
# HYPERTENSION
# ---------------------------------------------------------------------------
elif page == "Hypertension":
    with st.form("hypertension_form"):
        cols_meta = HYPERTENSION_COLUMNS
        values = {}
        medication_options = ["ACE Inhibitor", "Beta Blocker", "Diuretic", "Other", "None"]
        for col in cols_meta:
            if col == "BP_History":
                values[col] = st.selectbox(col, ["Normal", "Prehypertension", "Hypertension"])
            elif col == "Exercise_Level":
                values[col] = st.selectbox(col, ["Low", "Moderate", "High"])
            elif col == "Family_History":
                values[col] = st.selectbox(col, ["NO", "Yes"])
            elif col == "Smoking_Status":
                values[col] = st.selectbox(col, ["Non-Smoker", "Smoker"])
            elif col == "Medication":
                values[col] = st.selectbox(col, medication_options)
            else:
                values[col] = st.number_input(col, value=0.0)
        submitted = st.form_submit_button("Predict")
 
    if submitted:
        try:
            cols = HYPERTENSION_COLUMNS
            ordered_row = build_ordered_row(values, cols)
            input_df = pd.DataFrame([ordered_row], columns=cols)
            medication_map = {"ACE Inhibitor": 0, "Beta Blocker": 1, "Diuretic": 2, "Other": 3, "None": 4}
            input_df['BP_History'] = replace_values_in_csv(input_df, 'BP_History', "Normal", "Prehypertension", "Hypertension")
            input_df['Exercise_Level'] = replace_values_in_csv(input_df, 'Exercise_Level', "Low", "Moderate", "High")
            input_df['Family_History'] = replace_values_in_csv(input_df, 'Family_History', "NO", "Yes", None)
            input_df['Smoking_Status'] = replace_values_in_csv(input_df, 'Smoking_Status', "Non-Smoker", "Smoker", None)
            input_df['Medication'] = input_df['Medication'].map(medication_map).astype(int)
            scaled_cols = ['Age', 'Salt_Intake', 'Stress_Score', 'BP_History', 'Sleep_Duration', 'BMI', 'Medication', 'Exercise_Level']
            final_input = scalingfortest(input_df, scaled_cols, M["hyper_scaler"], None)
            proba = float(M["hyper_model"].predict(final_input, verbose=0)[0][0])
            risk_score = proba * 100
            confidence_pct = round((proba if proba >= 0.5 else (1 - proba)) * 100, 1)
            message, recommendation = riskscore_messege(risk_score)
            level, level_text = leveltext_predict(risk_score)
            render_result(round(risk_score, 1), level, level_text, message, recommendation, confidence_pct, risk_score)
        except Exception as e:
            st.error(f"Error: {e}")
