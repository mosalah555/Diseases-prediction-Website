from flask import Flask ,request ,jsonify ,render_template
import joblib
import pathlib 
from  models_files.utils import *
from sklearn.preprocessing import StandardScaler
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

BASE_DIR_FLASK = pathlib.Path(__file__).resolve().parent  
FRONTEND_DIR = BASE_DIR_FLASK.parent / "frontend"                
app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),  
    static_folder=str(FRONTEND_DIR),      
    static_url_path="/static"
)
@app.route("/") 
def index():
    return render_template("index.html")
@app.route("/api/predict/heart", methods=["POST"])
def heart_prediction():
    values = request.get_json(silent=True) or {}
    systolic_bp = float(values["systolic_bp"])
    diastolic_bp = float(values["diastolic_bp"])
    heart_rate = float(values["heart_rate"])
    cholesterol_mg_dl = float(values["cholesterol_mg_dl"])
    smoking_status = float(values["smoking"])
    alcohol_consumption = float(values["alcohol_consumption"])
    physical_activity = float(values["physical_activity"])
    bmi = float(values["bmi"])
    Map = MAP(systolic_bp ,diastolic_bp)
    Rpp = RPP(systolic_bp ,heart_rate)
    Pp = PP(systolic_bp ,diastolic_bp)
    unhealthy_lifestyle_score = UnhealthyLifeScore(smoking_status ,alcohol_consumption ,physical_activity)
    atherogenic_index_coefficient = AtherogenicIndexCoefficient(cholesterol_mg_dl ,systolic_bp)
    smoking_hypertension_interaction = SmokingHypertensionInteraction(smoking_status ,systolic_bp)
    cardiac_adiposity_proxy = CardiacAdiposityProxy(bmi ,heart_rate)
    cardiovascular_stress_index = CardiovascularStressIndex(Map ,heart_rate)

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
    orderd_row = []
    for col in cols:
        if col not in full_values:
            raise ValueError(f"Missing value for : {col}")
        orderd_row.append(float(full_values[col]))
    input = pd.DataFrame([orderd_row] ,columns=cols)
    scale_cols = ['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp',
              'diastolic_bp', 'bmi', 'MAP',
              'RPP Rate Pressure Product', 'PP Pulse Pressure',
              'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction',
              'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index'

               ]
    unimportant_cols = ['gender' ,'alcohol_consumption' ,'heart_rate']
    final_input = scalingfortest(input ,scale_cols ,heart_scaler ,unimportant_cols)
    proba = float(heart_model.predict(final_input)[0][0])
    risk_score = proba * 100
    confidence_pct = round((proba if proba >= 0.5 else (1 - proba)) * 100, 1)                
    message ,recommendation = riskscore_messege(risk_score)
    if risk_score >= 66:
        level, level_text = "high", "High Risk"
    elif risk_score >= 33:
        level, level_text = "moderate", "Moderate Risk"
    else:
        level, level_text = "low", "Low Risk"

    return jsonify({
        "riskScore": round(risk_score, 1),
        "level": level,
        "levelText": level_text,
        "detail": message,
        "message": message,
        "recommendation": recommendation,
        "confidencePct": confidence_pct,
        "predictedLabel": None,
    })
'''
@app.route("/api/predict/kidney", methods=["POST"])
def kidney_prediction():
    values = request.get_json(silent=True) or {}
    cols = KIDNEY_COLUMNS
    ordered_row = []
    for col in cols:
        if col not in values:
            raise ValueError(f"Missing value for : {col}")
        ordered_row.append(values[col])
    input = np.array([ordered_row])


@app.route("/api/predict/anemia", methods=["POST"])
def anemia_prediction():
    values = request.get_json(silent=True) or {}


@app.route("/api/predict/diabetes", methods=["POST"])
def diabetes_prediction():



@app.route("/api/predict/stroke", methods=["POST"])




@app.route("/api/predict/liver", methods=["POST"])
'''


'''
cd backend
$env:FLASK_APP = "app.py"
flask run
'''
