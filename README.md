# Hast du Alzheimer?
## DS23 - Projektmanagement - Praxisprojekt bei Valentin Zacharias --> Abfrage zur Einschätzung für eine Alzheimer-Wahrscheinlichkeit

# Ziel:
Das Ziel des Projektes ist es mittels eines Datensatzes von Kaggle ein Tool zu bauen um die Gefährdung einer Erkrankung an Alzheimer einzuschätzen. </br></br></br></br>

# Datensatz:
https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global/data

# Hugging-Face-Projekt:
https://huggingface.co/DS23-KI-Projekt </br></br></br></br>

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
# Installtion von requirements.txt
```
!pip install -r "https://raw.githubusercontent.com/DSKI23/alzheimervalentin/refs/heads/main/requirements.txt"
```
</br></br></br></br>
# Verzeichnisstruktur:
## Vorbereitung
In diesem Verzeichnis sind zwei Jupyter Notebooks, welche den Upload des Dataset nach https://huggingface.co/datasets/DS23-KI-Projekt/alzheimerdataset

## DataExploration
Hier werden Notebooks zur Visualisierung und Untersuchung des Datensatzes angelegt.

## DataCleaning
Bereinigung der Daten, sowie den Split (randomseed42) des Datasets und anschließenden Upload nach https://huggingface.co/datasets/DS23-KI-Projekt/alzheimerdataset_split beschreiben. 

## Modell
ML-Modell (Random Forest Classifier)
