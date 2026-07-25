from flask import Flask
import joblib
import pathlib
from  models_files.utils import *
BASE_DIR = pathlib.Path(__file__).resolve().parent / "saved model"
heart_model = joblib.load(BASE_DIR / "heart model.joblib")
heart_scaler = joblib.load(BASE_DIR / "heart_scaler.joblib")
kidney_model = joblib.load(BASE_DIR / "kidney model.joblib")
kidney_scaler = joblib.load(BASE_DIR / "kidney_scaler.joblib")
liver_model = joblib.load(BASE_DIR / "liver model.joblib")
liver_scaler = joblib.load(BASE_DIR / "liver_scaler.joblib")
sroke_model = joblib.load(BASE_DIR / "stroke model.joblib")
stroke_scaler = joblib.load(BASE_DIR / "stoke_scaler.joblib")
diabetes_model = joblib.load(BASE_DIR / "diabetes model.joblib")
diabetes_scaler = joblib.load(BASE_DIR / "diabetes_scaler.joblib")
anemia_model = joblib.load(BASE_DIR / "anemia model.joblib")
anemia_encoder = joblib.load(BASE_DIR / "anemia_encoder.joblib")
anemia_scaler = joblib.load(BASE_DIR / "anemia_scaler.joblib")
HEART_COLUMNS = [
    "age", "gender", "glucose_mg_dl", "cholesterol_mg_dl", "systolic_bp",
    "diastolic_bp", "heart_rate", "alcohol_consumption", "smoking_status",
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
 
app = Flask(__name__, static_folder="frontend", static_url_path="/frontend")

@app.route("/api/predict/heart", methods=["POST"])



@app.route("/api/predict/kidney", method=["Post"])



@app.route("/api/predict/anemia", methods=["Post"])


@app.route("/api/predict/diabetes", methods=["Post"])


@app.route("/api/predict/stroke", methods=["Post"])


@app.route("/api/predict/liver", methods=["Post"])

