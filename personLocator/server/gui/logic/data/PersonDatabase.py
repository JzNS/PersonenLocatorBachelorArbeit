import json
import os
import numpy as np


class PersonDatabase:
    """
    Dieser gesamte code wird noch erneurt auf postgresql und nicht auf eine json, diese ist nur zum testen!
    """
    DB_FILE = "server/config/person_db.json"

    def __init__(self):
        self.known_persons = {}
        self.load()

    def load(self):
        """Lädt die bekannte Personen-Datenbank aus der JSON-Datei. Wenn die Datei nicht existiert, wird eine leere DB erstellt."""
        if os.path.exists(self.DB_FILE):
            try:
                with open(self.DB_FILE, 'r') as f:
                    self.known_persons = json.load(f)
            except:
                self.known_persons = {}
        os.makedirs(os.path.dirname(self.DB_FILE), exist_ok=True)

    def save(self):
        """Speichert die aktuelle Personen-Datenbank in der JSON-Datei."""
        try:
            with open(self.DB_FILE, 'w') as f:
                json.dump(self.known_persons, f, indent=4)
        except Exception as e:
            print(f"Fehler beim Speichern der DB: {e}")

    def get_person_data(self, name):
        return self.known_persons.get(name)

    def register_person(self, name, height, avg_colors):
        self.known_persons[name] = {
            "height": height,
            "colors": avg_colors,
            "last_seen": 0
        }
        self.save()
        print(f"DB: Person '{name}' gespeichert.")

    def update_person(self, name, data):
        if name not in self.known_persons:
            self.known_persons[name] = {}
        for k, v in data.items():
            self.known_persons[name][k] = v
        self.save()

    def match_person(self, current_height, width=None, colors=None):
        """Findet die beste Übereinstimmung basierend auf der Größe (und optionalen Farben)."""
        best_match = None
        best_score = 0.0

        for name, data in self.known_persons.items():
            # Einfacher Größen-Check
            h_diff = abs(data.get("height", 175) - current_height)
            if h_diff > 10: continue

            score = 100 - h_diff
            if score > best_score and score > 80:
                best_match = name
                best_score = score

        return best_match