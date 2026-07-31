# Vitals Panel — Multi-Disease Risk Screening Web App

A full-stack web application that screens for **8 different health conditions** from a single interface, each powered by its own trained neural network. Built with a Flask backend serving TensorFlow/Keras models, and a vanilla HTML/CSS/JS frontend with no external frontend framework.

> ⚠️ **Disclaimer:** This is an educational / portfolio project. Not all training data is real-world clinical data — some datasets are synthetic or semi-synthetic and used strictly for learning purposes. **This tool does not replace clinical judgment, laboratory diagnosis, or professional medical advice**, and should not be used for real diagnostic decisions.

---

## What it does

Pick a condition from the tab bar, fill in the patient's values (or click "Fill sample values" to autofill a realistic example), and run the model. Each tab's form fields map 1:1 to the exact feature columns of that model's training dataset — nothing is guessed or reformatted on the way in.

The backend runs the real trained model for every request and returns a risk score, a plain-language message, a recommendation, and — for the two multi-class models — a specific predicted diagnosis label.
it can work with flask by flask_app.py or with streamlit by st_app.py
---

## Models included

| # | Condition | Model type | Test Accuracy | Test Recall | Test Loss |
|---|---|---|---|---|---|
| 1 | Heart Attack | Binary (NN) | 89.00% | 94.12% | 0.3593 |
| 2 | Anemia | Multi-class, 9 diagnoses (NN) | 94.82% | 92.23% | 0.0837 |
| 3 | Stroke | Binary (NN) | 93.45% | **0.00%** ⚠️ | 0.6410 |
| 4 | Kidney Disease (risk tier) | Multi-class, 3 tiers (NN) | 89.61% | 89.61% | 0.4046 |
| 5 | Diabetes | Binary (NN) | 92.19% | 86.03% | 0.1547 |
| 6 | Liver Disease | Binary (NN) | 69.49% | 68.89% | 0.6089 |
| 7 | Hypertension | Binary (NN) | 91.96% | 93.72% | 0.2498 |
| 8 | Chronic Kidney Disease (CKD) | Binary (NN) | **100.00%** ⚠️ | **100.00%** ⚠️ | 0.0882 |
| — | Alzheimer's | Binary (NN) | 89.77% | 80.00% | 0.4089 |

### ⚠️ Known limitations (documented honestly, not hidden)

- **Stroke model: 0% recall.** Despite 93% accuracy, the model has a recall of exactly zero on the test set — this almost always means the model is simply predicting the majority class ("no stroke") every time, driven by severe class imbalance in the training data. Accuracy alone is misleading here; this model needs class-weighting, resampling (e.g. SMOTE), or a lower decision threshold before it can be considered usable.
- **Chronic Kidney Disease model: 100% across the board.** A perfect score on every metric on a small dataset (400 rows) is a strong signal of overfitting or test/train leakage rather than genuine generalization — this result should not be trusted at face value without further validation on unseen data.
- **Liver Disease model** has the weakest overall performance (69% accuracy) among the binary models, suggesting the current feature set/dataset size may not be sufficient for reliable predictions.
- Datasets vary significantly in size and realism; some are used purely to demonstrate the end-to-end pipeline (data → preprocessing → model → API → UI), not to produce clinically valid predictions.

---

## Architecture

```
Browser (HTML/CSS/JS)
        │  fetch() POST JSON
        ▼
Flask backend (app.py)
        │  loads all 8 models + scalers + encoders once at startup
        ▼
scikit-learn StandardScaler → TensorFlow/Keras model.predict()
        │
        ▼
JSON response → rendered inline in the same page
```

- **Frontend:** static HTML/CSS/JS, no build step, no framework. One shared page with a tab per disease; form fields and API calls are both driven by a single JS config array so adding a new disease is a matter of adding one object to that array plus one Flask route.
- **Backend:** Flask, one `/api/predict/<model_id>` route per disease. Each route re-encodes categorical inputs exactly as done in training, applies the matching `StandardScaler`, and runs the corresponding Keras model.
- **Models:** trained separately per disease with TensorFlow/Keras (`Sequential` + `Dense` + `Dropout`, `EarlyStopping`, class-weighting for imbalanced targets), saved via `joblib`.

---

## API contract

Every model shares one response shape:

```json
POST /api/predict/<model_id>
Content-Type: application/json

{
  "riskScore": 71.4,
  "level": "high",
  "levelText": "High Risk",
  "detail": "Additional context for this result.",
  "message": "The indicators suggest a high risk.",
  "recommendation": "A medical consultation is recommended as soon as possible.",
  "confidencePct": 78.2,
  "predictedLabel": "Iron deficiency anemia"
}
```

`predictedLabel` is only populated for the multi-class models (Anemia, Kidney Disease); it's `null`/omitted for the binary ones.

`model_id` is one of: `heart`, `anemia`, `stroke`, `kidney`, `diabetes`, `liver`, `hypertension`, `chronic_kidney`.

---
##AI USES IN THE APP
1_Used to make suggestions to improve the models training process
2_helped me in the pathes
3_Helped me in some error analysis and fixing
4_Helped me in making st_app.py from the flask_app.py that i have made by my hand
##Note :
ALL THE LOGIC OF THE APP IS FROM MY MADE 
## Tech stack

- **Backend:** Python, Flask, TensorFlow/Keras, scikit-learn, pandas, NumPy, joblib
- **Frontend:** HTML5, CSS3, vanilla JavaScript (no framework, no build tools)
- **Data:** a mix of public health datasets (Kaggle-sourced) and synthetic data, used for educational purposes

---

## Running it locally

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in a browser.

---

## Project structure

```
Diseases prediction model/
├── backend/
│   ├── app.py                    # Flask routes, one per disease
│   ├── models_files/
│   │   ├── utils.py               # shared preprocessing helpers
│   │   └── *_prediction_model.py   # training script per disease
│   └── saved model/
│       └── *.joblib                # trained models, scalers, encoders
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

---

## License

See `LICENSE` file in the repository.
