import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# 1. Load processed data

df = pd.read_csv('train_landmarks.csv', header = None)

X = df.iloc[:,:-1].values #All columns excenpt the last one (landmarks)

y = df.iloc[:,-1].values #The last column labels

# 2. encode labels into numbers (Mudra names -> 0,1,2,...)
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# 3. Build a simple dense neural network

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy',metrics =['accuracy'])

# 4. train
model.fit(X, y_encoded, epochs=50, batch_size = 32)

# 5. Convert to TFlite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('bharatnatyam_mudra.tflite','wb') as f:
    f.write(tflite_model)
    print("Custom model saved as bharatnatyam_mudra.tflite")

