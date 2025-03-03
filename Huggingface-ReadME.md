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

Diese Anwendung ist ein interaktives Screening-Tool für die Risikobewertung von Alzheimer, basierend auf einem Random Forest Modell.

## Funktionen
- Benutzerfreundliche Oberfläche zur Eingabe von Patientendaten
- Echtzeit-Risikobewertung mit maschinellem Lernen
- Farbcodierte Risikolevel und Konfidenzwerte
- Anpassbare Empfehlungen basierend auf der Bewertung

## Technische Umsetzung
- Python mit scikit-learn für Modellvorhersagen
- Gradio für die Weboberfläche
- Direkte Integration des Pickle-Modells
- Numpy für effiziente Datenverarbeitung

## Hinweis für Entwickler
Das `.pkl`-Modell muss im Hauptverzeichnis vorhanden sein und erwartet Eingabedaten im Format:
- Alter (20-110)
- BMI (15-40)
- Bildungsniveau (0-19 Jahre)
- Geschlecht (M/F)
- Familiäre Vorbelastung (J/N)
- Kognitiver Testwert (30-99)
- Genetischer Risikofaktor (J/N)

## Disclaimer
Dieses Tool dient nur zu Screening-Zwecken und sollte nicht als diagnostisches Instrument verwendet werden.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference