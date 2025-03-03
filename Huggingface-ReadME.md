---
title: Alzheimers Screening Assistant
emoji: 📉
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.19.0
app_file: app.py
pinned: false
short_description: Demo for user interfacing and screening assistant homepage
---

# Alzheimer's Disease Screening Assistant

## Übersicht
Dieser Hugging Face Space bietet eine interaktive Weboberfläche für das Alzheimer-Risikoscreening basierend auf einem trainierten Random Forest Modell. Der Assistent nimmt Patienteninformationen als Eingabe entgegen und liefert eine Risikobewertung zusammen mit personalisierten Empfehlungen.

## Funktionen
- **Benutzerfreundliche Oberfläche**: Einfache Schieberegler und Buttons zur Eingabe von Patientendaten
- **Echtzeit-Bewertung**: Sofortige Risikobewertung mittels maschinellem Lernen
- **Visuelle Rückmeldung**: Farbcodierte Risikostufen und Konfidenzwerte
- **Personalisierte Empfehlungen**: Maßgeschneiderte Vorschläge basierend auf den Bewertungsergebnissen
- **Integration mit ML-Modell**: Direkte Integration mit einem Random Forest Klassifikator

## Funktionsweise
1. **Datenerfassung**: Die Oberfläche sammelt wichtige Patienteninformationen:
   - Alter (20-110)
   - BMI (15-40)
   - Bildungsniveau (0-19 Jahre)
   - Geschlecht (Männlich/Weiblich)
   - Familiäre Vorgeschichte von Alzheimer (Ja/Nein)
   - Kognitiver Testwert (30-99)
   - Genetischer Risikofaktor (Ja/Nein)
2. **ML-Integration**: Die Anwendung lädt ein vortrainiertes Random Forest Modell (`random_forest_alzheimer.pkl`), das die Eingabedaten verarbeitet.
3. **Risikobewertung**: Das Modell liefert eine binäre Klassifikation (0=Niedriges Risiko, 1=Hohes Risiko) zusammen mit einem Konfidenzwert.
4. **Empfehlungen**: Basierend auf der Bewertung gibt das System maßgeschneiderte Empfehlungen für den Patienten.

## Technische Implementierung
- **Backend**: Python mit scikit-learn für Modellvorhersagen
- **Frontend**: Gradio für die Weboberfläche
- **Modellintegration**: Direktes Laden der Pickle-Modelldatei
- **Datenverarbeitung**: Numpy für effiziente Datenverarbeitung

## Hinweise für Entwickler
- Die `.pkl`-Modelldatei muss im Stammverzeichnis vorhanden sein, es wird mit actions auf dem neusten stand gebracht
- Das Modell erwartet ein spezifisches Eingabeformat, wie in der Benutzeroberfläche definiert
- Die Ausgabe bietet sowohl benutzerfreundliche visuelle Ergebnisse als auch technische Details
- Fehlerbehandlung ist für das Laden des Modells und Vorhersagefehler implementiert

## Haftungsausschluss
Dieses Tool ist nur für Screening-Zwecke konzipiert und sollte nicht als diagnostisches Instrument verwendet werden. Konsultieren Sie immer medizinisches Fachpersonal für eine ordnungsgemäße Diagnose und Behandlung von Alzheimer.

---
*Diese Anwendung ist Teil eines größeren Projekts. Für weitere Informationen über das Modelltraining und die Entwicklung besuchen Sie bitte das Hauptrepository https://huggingface.co/DS23-KI-Projekt

Schauen Sie sich die Konfigurationsreferenz unter https://huggingface.co/docs/hub/spaces-config-reference an