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

df = pd.read_csv("C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Diseases prediction model\\backend\\dataset\\Indian Liver Patient Dataset (ILPD).csv")

df['gender'] = replace_values_in_csv(df ,'gender' ,"Female" ,"Male" ,None)

X = df.drop(columns=['is_patient'])
Y = df['is_patient'].map({1: 1, 2: 0})
print(X.isnull().sum())
print(np.isinf(X.select_dtypes(include=[np.number])).sum())

X['alkphos'] = X['alkphos'].fillna(X['alkphos'].median())
x_train ,x_val ,x_test ,y_train ,y_val ,y_test = data_splitting(X ,Y ,0.8 ,0.1 ,0.1 ,42)

scaler = StandardScaler()
scaled_cols = ['age' ,'tot_bilirubin' ,'direct_bilirubin' ,'tot_proteins' ,'albumin' ,'ag_ratio' ,'sgpt' ,'sgot' ,'alkphos']
final_x_train = scalingfortrain(x_train ,scaled_cols ,scaler ,None)
final_x_val = scalingfortest(x_val ,scaled_cols ,scaler ,None)
final_x_test = scalingfortest(x_test ,scaled_cols ,scaler ,None)

model = Sequential([
    Dense(16 ,input_shape=(10,) ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l1'),
    Dropout(0.02),
    Dense(12 ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l2'),
    Dropout(0.02),
    Dense(8 ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l3'),
    Dropout(0.02),
    Dense(4 ,activation='relu' ,kernel_regularizer=l2(0.001) ,name='l4'),
    Dropout(0.02),
    Dense(1 ,activation='sigmoid' ,name='l5')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy' , tf.keras.metrics.Recall()]
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
)

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights = dict(enumerate(weights))
history = model.fit(
    final_x_train ,y_train,
    validation_data=(final_x_val ,y_val),
    epochs=250,
    class_weight=weights,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)

print('Model Evaluation on test data :     ')
test_loss ,test_accuracy ,test_recall = model.evaluate(final_x_test ,y_test)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Recall : {test_recall:.4f}")

BASE_DIR = pathlib.Path("Diseases prediction model").resolve().parent.parent
model_path = BASE_DIR /"saved model" / "liver model.joblib"
joblib.dump(model, model_path)
print(f"the model has saved in: {model_path}")
joblib.dump(scaler, BASE_DIR / "saved model" / "liver_scaler.joblib")


