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
df = pd.read_csv("C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Diseases prediction model\\backend\\dataset\\chronic kidney disease-CKD.csv")
X = df.drop(columns=['Class'])
Y = df['Class']
x_train ,x_val ,x_test ,y_train ,y_val ,y_test = data_splitting(X ,Y ,0.7 ,0.2 ,0.1 ,42)
scaler = StandardScaler()
scaled_cols = ['Bp' ,'Sg' ,'Al' ,'Su' ,'Pot', 'Bu' ,'Sc' ,'Sod' ,'Hemo' ,'Wbcc' ,'Rbcc']
final_x_train = scalingfortrain(x_train ,scaled_cols ,scaler ,None)
final_x_val = scalingfortest(x_val ,scaled_cols ,scaler ,None)
final_x_test = scalingfortest(x_test ,scaled_cols ,scaler ,None)
model = Sequential([
    Dense(13 ,input_shape=(final_x_train.shape[1],) ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l1'),
    Dropout(0.02),
    Dense(9 , activation='relu' ,kernel_regularizer=l2(0.01) ,name='l2'),
    Dropout(0.02),
    Dense(5 ,activation='relu' ,kernel_regularizer=l2(0.01) ,name='l3'),
    Dropout(0.02),
    Dense(1 ,activation='sigmoid' ,name='l4')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy' ,tf.keras.metrics.Recall()]
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights = dict(enumerate(weights))
history = model.fit(
    final_x_train ,y_train ,
    validation_data=(final_x_val ,y_val),
    epochs=250,
    class_weight=weights,
    batch_size=16,
    callbacks=[early_stop],
    verbose=2
) 
print("Model Evaluation on test data :   ")
test_loss ,test_accuracy ,test_recall = model.evaluate(final_x_test ,y_test)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Recall : {test_recall:.4f}")

BASE_DIR = pathlib.Path("Diseases prediction model").resolve().parent.parent 
model_path = BASE_DIR   / "saved model" / "chronic model.joblib"
joblib.dump(model ,model_path)
joblib.dump(scaler ,BASE_DIR / "saved model" / "chronic_scaler.joblib")


