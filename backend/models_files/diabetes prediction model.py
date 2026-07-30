import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from utils import *
import pathlib
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError ,BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
values ={
    "never":0,
    "not current":1,
    "current":3,
    "former":2,
    "ever":1,
    "No Info":1
}
#get the dataset from the device Note: you must to change it before work to the place of it on your device
df = pd.read_csv("C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Diseases prediction model\\backend\\dataset\\diabetes_prediction_dataset.csv")
#Replacing the strings value to numeric 0/1/2
df['gender'] = replace_values_in_csv(df ,'gender' ,"Female" ,"Male" ,"Other")
df['smoking_history'] = df['smoking_history'].replace(values).astype(int)
#defining X and Y ,Splitting the data
X = df.drop(columns=['diabetes'])
Y = df['diabetes']
x_train ,x_val ,x_test ,y_train ,y_val ,y_test = data_splitting(X ,Y ,0.7 ,0.2 ,0.1 ,42)
#Scaling the data
scaler = StandardScaler()
scaled_cols = ['age' ,'smoking_history' ,'bmi' ,'HbA1c_level' ,'blood_glucose_level']
final_x_train = scalingfortrain(x_train ,scaled_cols ,scaler ,None)
final_x_valid = scalingfortest(x_val ,scaled_cols ,scaler ,None)
final_x_test = scalingfortest(x_test ,scaled_cols ,scaler ,None)
#Dfining the model
model = Sequential([
    Dense(16 ,input_shape=(final_x_train.shape[1],) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l1'),
    Dropout(0.08),
    Dense(8 ,input_shape=(16,) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l2'),
    Dropout(0.06),
    Dense(4 ,input_shape=(8,) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l3'),
    Dropout(0.04),
    Dense(1 ,input_shape=(4,) ,activation='sigmoid')
])
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
  metrics=['accuracy' ,tf.keras.metrics.Recall()]
 )
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)
#Training the model
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights = dict(enumerate(weights))
history = model.fit(
    final_x_train ,y_train ,
    validation_data=(final_x_valid ,y_val),
    epochs=500,
    class_weight=weights,
    batch_size=128 ,
    callbacks=[early_stop] ,
    verbose=2
)
#Saving the model
print("Model evaluation on test data :  ")
test_loss ,test_accuracy ,test_recall = model.evaluate(final_x_test ,y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent / "saved model"
model_path = BASE_DIR / "diabetes model.joblib"
joblib.dump(model , model_path)
joblib.dump(scaler , BASE_DIR / "diabetes_scaler.joblib")
print(f"Model saved successfully to {model_path}")