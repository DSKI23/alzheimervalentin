from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import pickle 

state = int(os.environ['state'])

df = pd.read_pickle("dataset.pkl") 
os.remove("dataset.pkl")

target = 'Alzheimer’s Diagnosis'
features = df.columns.tolist()
features.remove(target)

X = df[features]
y = df[target]

# Daten in Trainings- und Testset aufteilen 80 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state, stratify=y)

# Random Forest initialisieren
rf_model = RandomForestClassifier(n_estimators=50, random_state=state)

# Modell trainieren
rf_model.fit(X_train, y_train)

# Vorhersagen auf dem Testset
y_pred = rf_model.predict(X_test)




with open('random_forest_alzheimer.pkl', 'wb') as file:
    pickle.dump(rf_model, file)