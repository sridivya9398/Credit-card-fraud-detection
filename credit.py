import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
# Load data
df = pd.read_csv('/Users/spirisingula@unomaha.edu/Downloads/creditcard.csv')

# Basic data cleaning and preprocessing
df.dropna(inplace=True)

# Feature and target separation
X = df.drop('Class', axis=1)  # Assuming 'Class' is the column for fraud/not fraud
y = df['Class']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE()
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_sm, y_train_sm)

predictions = clf.predict(X_test_scaled)
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
