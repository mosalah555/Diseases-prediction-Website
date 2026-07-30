import numpy as np
import pandas as pd
import tensorflow 
import joblib
import pathlib
from utils import *
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Diseases prediction model\\backend\\dataset\\healthcare-dataset-stroke-data.csv")

df['bmi'] = df['bmi'].fillna(df['bmi'].median())

df['gender'] = replace_values_in_csv(df ,'gender' ,"Female" ,"Male" ,"Other")
df['ever_married'] = replace_values_in_csv(df ,'ever_married' ,"No" ,"Yes" ,None)
df['Residence_type'] = replace_values_in_csv(df ,'Residence_type' ,"Rural" ,"Urban" ,None)
smoking_map = {
    "never smoked":0 ,
    "formerly smoked":1 ,
    "Unknown":1 ,
    "smokes":2
}
df['smoking_status'] = df['smoking_status'].map(smoking_map).astype(int)
work_type_map = {
    "Private": 0,
    "Self-employed": 1,
    "Govt_job": 2,
    "children": 3,
    "Never_worked": 4
}
df['work_type'] = df['work_type'].map(work_type_map).astype(int)
X = df.drop(columns=['stroke'])
Y = df['stroke']

x_train ,x_val ,x_test ,y_train ,y_val ,y_test = data_splitting(X ,Y ,0.7 ,0.2 ,0.1 ,42)

scaler = StandardScaler()
scaled_cols = ['age' ,'work_type' ,'avg_glucose_level' ,'bmi' ,'smoking_status']
final_x_train = scalingfortrain(x_train ,scaled_cols ,scaler ,None)
final_x_val =  scalingfortest(x_val ,scaled_cols ,scaler ,None)
final_x_test = scalingfortest(x_test ,scaled_cols ,scaler ,None)

model = Sequential([
    Dense(10 ,input_shape=(final_x_train.shape[1],) ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l1'),
    Dropout(0.02),
    Dense(8 ,input_shape=(10,) ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l2'),
    Dropout(0.02),
    Dense(6 ,input_shape=(8,) ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l3'),
    Dropout(0.02),
    Dense(4 ,input_shape=(32,) ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l4'),
    Dropout(0.02),
    Dense(1 ,input_shape=(4,) ,activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy' ,tf.keras.metrics.Recall()]
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights = dict(enumerate(weights))
history = model.fit(
    final_x_train ,y_train ,
    validation_data=(final_x_val ,y_val),
    epochs=100,
    class_weight=weights,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2
)

print("Model Evaluation on test data :   ")
test_loss ,test_accuracy ,test_recall = model.evaluate(final_x_test ,y_test)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall : {test_recall:.4f}")

BASE_DIR = pathlib.Path("Diseases prediction model").resolve().parent.parent  / "Diseases prediction model"
model_path = BASE_DIR / "backend" / "saved model" / "stroke model.joblib"
joblib.dump(model , model_path)
joblib.dump(scaler ,BASE_DIR / "backend" / "saved model" / "stoke_scaler.joblib")



