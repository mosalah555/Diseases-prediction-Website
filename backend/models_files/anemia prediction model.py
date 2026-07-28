import numpy as np
import pandas as pd
import tensorflow 
import joblib
from utils import *
import pathlib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Diseases prediction model\\backend\\dataset\\diagnosed_cbc_data_v4.csv")
print(df['Diagnosis'].value_counts())
X = df.drop(columns=['Diagnosis'])
Y = df['Diagnosis']
y_forweight = Y
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
Y = to_categorical(Y_encoded ,num_classes=9)
x_train ,x_val ,x_test ,y_train ,y_val ,y_test = data_splitting(X ,Y ,0.75 ,0.15 ,0.1 ,42)
scaler = StandardScaler()
scaled_cols = ['WBC' ,'LYMp' ,'NEUTp' ,'LYMn' ,'NEUTn' ,'RBC' ,'HGB' ,'HCT' ,'MCV' ,'MCH' ,'MCHC' ,'PLT' ,'PDW' ,'PCT']
final_x_train = scalingfortrain(x_train ,scaled_cols ,scaler ,None)
final_x_val = scalingfortest(x_val ,scaled_cols ,scaler ,None)
final_x_test = scalingfortest(x_test ,scaled_cols ,scaler ,None)

model = Sequential([
    Dense(14 ,input_shape=(final_x_train.shape[1],) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l1'),
    Dropout(0.02),
    Dense(12 ,input_shape=(14,) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l2'),
    Dropout(0.02),
    Dense(10 ,input_shape=(12,) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l3'),
    Dropout(0.02),
    Dense(9 ,input_shape=(10,) ,activation='softmax' ,name='l5')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy' ,tf.keras.metrics.Recall()]
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
#weights = compute_class_weight('balanced', classes=np.unique(y_forweight), y=y_forweight)
#weights = dict(enumerate(weights))
history = model.fit(
    final_x_train ,y_train,
    validation_data=(final_x_val ,y_val),
    epochs=250,
    #class_weight=weights,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)

print("Model Evaluation on test data :   ")
test_loss ,test_accuracy ,test_recall = model.evaluate(final_x_test ,y_test)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Recall : {test_recall:.4f}")

BASE_DIR = pathlib.Path("Diseases prediction model").resolve().parent.parent 
model_path = BASE_DIR   / "saved model" / "anemia model.joblib"
joblib.dump(model ,model_path)
joblib.dump(scaler ,BASE_DIR / "saved model" / "anemia_scaler.joblib")
joblib.dump(label_encoder ,BASE_DIR / "saved model" / "anemia_encoder.joblib")

