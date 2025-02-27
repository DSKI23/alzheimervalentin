from sklearn.ensemble import RandomForestClassifier
import huggingface_hub
import joblib
import os

state = os.environ('state')
huggingface_hub.login(token= os.environ['huggingface_token'])
rf_model = RandomForestClassifier(n_estimators=50, random_state=state)

# Modell speichern .pkl-Datei
model_filename = "random_forest_alzheimer.pkl"
joblib.dump(rf_model, model_filename)

# Repository-Name auf Hugging Face (anpassen!)
hf_repo_name = "DS23-KI-Projekt/KI-Modell"

# Falls das Repository nicht existiert, erstelle es
api = huggingface_hub.HfApi()
api.create_repo(repo_id=hf_repo_name, exist_ok=True)

# Modell hochladen
api.upload_file(
    path_or_fileobj=model_filename,  # Die gespeicherte Modell-Datei
    path_in_repo=model_filename,  # Name im Hugging Face Repo
    repo_id=hf_repo_name,  # Dein Repository
    commit_message="Upload des Random Forest Modells für Alzheimer-Diagnose"
)