import time
import numpy as np
from typing import Dict, List


class ServerPersonTracker:
    """
    Verwaltet die Fusionierung.
    Modus: NUR INFORMATION (Keine Berechnung, keine Offsets).
    Fix: Übergibt jetzt korrekte Argumente an GlobalPerson.
    """

    def __init__(self, person_database, person_class_ref):
        self.db = person_database
        self.PersonClass = person_class_ref

        self.global_persons = []
        self.next_id = 1
        self.input_buffer: Dict[str, list] = {}

    def update_camera_data(self, cam_name: str, person_list: list):
        self.input_buffer[cam_name] = person_list
        self.__process_fusion()

    def get_raw_tracks(self):
        return self.input_buffer

    def force_merge_and_calibrate(self, name: str, assignments: List[dict]):
        """
        Wird aufgerufen, wenn im Wizard 'Zuweisen' geklickt wird.
        Macht KEINE mathematische Fusion, sondern speichert nur Info in DB.
        """
        print(f"\n{'#' * 60}")
        print(f"[MANUELLE ZUWEISUNG] Name: '{name}'")
        print(f"{'#' * 60}")

        heights = [a.get('height', 175) for a in assignments]
        avg_height = sum(heights) / len(heights) if heights else 175.0

        self.db.register_person(name, avg_height, [])
        print(f"-> Person '{name}' in Datenbank aktualisiert (Höhe: {avg_height:.1f}cm).")

        print(f"-> Zugeordnete Clients:")
        for item in assignments:
            cam = item.get('cam')
            pid = item.get('id')
            pos = item.get('pos')
            print(f"   - {cam} (ID {pid}) an Position {pos}")

        print(f"{'#' * 60}\n")

        return f"Zuweisung für '{name}' gespeichert (Keine Verschiebung berechnet)."

    def __process_fusion(self):
        """
        Standard Logik: Zeigt Punkte an.
        FIX: Übergibt jetzt ID und Keypoints korrekt an GlobalPerson.update()
        """
        now = time.time()
        self.global_persons = [p for p in self.global_persons if now - p.last_update < 1.0]

        for cam_name, persons in self.input_buffer.items():
            for p_data in persons:
                if 'pos' not in p_data: continue

                raw_pos = np.array(p_data['pos'], dtype=np.float32)
                height = p_data.get('height', 175.0)
                client_id = p_data.get('id', 0)
                keypoints = p_data.get('keypoints', [])
                matched = False
                for gp in self.global_persons:
                    dist = np.linalg.norm(gp.pos - raw_pos)

                    if dist < 80.0:
                        gp.update(
                            new_pos=raw_pos,
                            new_height=height,
                            cam_name=cam_name,
                            client_id=client_id,
                            raw_pos=raw_pos,
                            keypoints=keypoints,
                            raw_data=p_data
                        )
                        matched = True
                        break

                if not matched:
                    new_gp = self.PersonClass(self.next_id, raw_pos, height)
                    new_gp.set_database(self.db)

                    known_name = self.db.match_person(height)
                    if known_name:
                        new_gp.load_identity(known_name)

                    new_gp.update(raw_pos, height, cam_name, client_id, raw_pos, keypoints,
                                  raw_data=p_data)

                    self.global_persons.append(new_gp)
                    self.next_id += 1