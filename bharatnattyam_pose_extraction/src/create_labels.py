# create_labels.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your training data again just to get the classes
df = pd.read_csv('train_landmarks.csv', header=None)
y = df.iloc[:, -1].values
encoder = LabelEncoder()
encoder.fit(y)

# Generate the labels file
with open('labels.txt', 'w') as f:
    for label in encoder.classes_:
        f.write(f"{label}\n")

print("labels.txt generated successfully!")