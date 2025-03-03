# Hast du Alzheimer?
## DS23 - Projektmanagement - Praxisprojekt bei Valentin Zacharias --> Abfrage zur Einschätzung für eine Alzheimer-Wahrscheinlichkeit

# Ziel:
Das Ziel des Projektes ist es mittels eines Datensatzes von Kaggle ein Tool zu bauen um die Gefährdung einer Erkrankung an Alzheimer einzuschätzen. </br></br></br></br>

# Datensatz:
https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global/data

# Hugging-Face-Projekt:
https://huggingface.co/DS23-KI-Projekt </br>
https://github.com/DSKI23/alzheimervalentin/blob/main/Huggingface-ReadME.md --> ReadME für die Gradio-Applikation bei HuggingFace</br></br></br></br>

# Projektmanagement / Jira Board
https://github.com/orgs/DSKI23/projects/1 
</br></br></br></br>
# Arbeit mit Google Colab
Um mit Google Colab zu arbeiten, muss in den Settings bei https://colab.research.google.com/ das Github-Konto verknüpft werden und der Haken bei "Auf private Repositories und Organisationen zugreifen" gesetzt werden:
![image](https://github.com/user-attachments/assets/138466f5-aa1a-4040-94f4-b3a36c95ce32)
</br></br></br></br>
# Import des gesplitteten Datasets in ein Notebook
```
!pip install --upgrade huggingface_hub
!pip install datasets

from huggingface_hub import notebook_login
from datasets import load_dataset

notebook_login()


dataset = load_dataset("DS23-KI-Projekt/alzheimerdataset_split")
dataset
```
</br></br></br></br>
# Installtion von requirements_colab.txt
Für die Ausführung der Notebooks mit google Colab wird die requirements_colab.txt-Datei benötigt.
```
!pip install -r "https://raw.githubusercontent.com/DSKI23/alzheimervalentin/refs/heads/main/requirements_colab.txt"
```
</br></br></br></br>
# Ausführungsreihenfolge:
1. Data Upload zu Huggingface mittels https://github.com/DSKI23/alzheimervalentin/blob/main/Vorbereitung/Upload_to_HF.ipynb
2. Bereinigung der Daten, sowie den Split mittels https://github.com/DSKI23/alzheimervalentin/blob/main/DataCleaning/data_cleaning_upload.ipynb
3. Training, Valuierung der Hyperparameter und Upload zu HuggingFace mittels https://github.com/DSKI23/alzheimervalentin/blob/main/Modell/training.ipynb
4. Gradio-Applikation: https://huggingface.co/spaces/DS23-KI-Projekt/alzheimers-screening-assistant
</br></br></br></br>
# Verzeichnisstruktur:
## Vorbereitung
In diesem Verzeichnis ist ein Jupyter-Notebook, welches den Upload des Dataset nach https://huggingface.co/datasets/DS23-KI-Projekt/alzheimerdataset vornimmt.

## DataExploration
Hier sind Notebooks zur Visualisierung und Untersuchung des Datensatzes angelegt.

## DataCleaning
Bereinigung der Daten, sowie den Split (randomseed42) des Datasets und anschließenden Upload nach https://huggingface.co/datasets/DS23-KI-Projekt/alzheimerdataset_split. 

## Modell
-->Philipp

## test
Hier wird der Datensatz auf Duplikate und None-Values überprüft.

## Hyperparameter Logging (über WandB)

1. WandB - Konto erstellen
Besuche die Website https://wandb.ai/site/models und erstelle unter SIGN UP einen Account, welches man für die Bearbeitung verwendet.
Den generierten API-Token sicher verwahren. Er wird beim Durchlaufen des Codes abgefragt und muss entsprechend eingefügt werden.

2. WandB Login
Mit folgendem Code wird die Anmeldung auf WandB durchgeführt: !wandb login
Beachte, dass für !wandb login auch der generierte API Token verwendet werden muss.

3. Lokalität von benötigtem Datensatz
Es gilt, den Dateipfad zu hinterlegen. Für unser aktuelles Projekt befindet sich der Datensatz unter: DS23-KI-Projekt/alzheimerdataset_split

Die folgenden Metriken werden zur Evaluation des Modells verwendet:

- Accuracy (Genauigkeit des Modells)
- Classification Report (Präzision, Recall, F1-Score)
- Confusion Matrix (Matrix zur Fehleranalyse)
   
4. Implementierung von Funktionen
Die Funktion load_data erstellt die Features (X) und Labels (y) aus dem Datensatz.
Die Funktion train_rf trainiert einen Random Forest Classifier.
Der Hyperparameter n_estimators gibt im Code die Anzahl der Bäume im Modell an.

In der main-Funktion sollten folgende Schritte abgebildet werden:
- Initialisierung des WandB-Projekts
- Laden des Datensatzes
- Aufteilung in Features (X) und Labels (y)
- Aufteilung des Datensatzes in Trainings- und Testdaten
- Training des Modells mit verschiedenen Hyperparametern
- Logging der Ergebnisse in WandB (Parameter und Hyperparameter)

5. Trainingsprozess
Das Modell wird mit variierendem n_estimators trainiert. Im Alzheimer Beispiel wählten wir testweise 10.

6. Speichern des Models
Das Modell mit der höchsten Accuracy wird gespeichert. Nutze hierfür den Befehl import pickle.

7. Visualisierung
Nach jedem Trainingsdurchlauf kann man in der Terminal Console die Links entnehmen, die eine Weiterleitung an die WandB Webseite mit verschiedenen Metriken darstellen.
