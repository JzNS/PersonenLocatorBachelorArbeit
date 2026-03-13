import time
import logging
import threading
import struct
import msgpack
from typing import Any, Optional, Dict

# Type-Hinting Import (verhindert Zirkelbezug zur Laufzeit)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client.network.ClientConnector import ClientConnector


class CommandSender:
    """
    Verantwortlich für das Erstellen und Versenden von Protokoll-Paketen.
    Nutzt MessagePack und Length-Prefixing (4-Byte Header) für maximale Performance.
    """
    PING_INTERVAL = 20

    def __init__(self, client_connector: 'ClientConnector', client_name: str) -> None:
        self.client_connector = client_connector
        self.client_name = client_name
        self.__is_running = True
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Startet den Hintergrund-Thread für Pings."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.__loop, daemon=True, name="PingThread")
        self._thread.start()
        logging.debug("Ping-Thread gestartet.")

    def stop(self) -> None:
        self.__is_running = False


    def send_register(self) -> None:
        """Initiale Anmeldung."""
        logging.info(f"Sende Registrierung für '{self.client_name}'...")
        self.send_message("SERVER", "REGISTER", {"name": self.client_name})

    def send_ping(self) -> None:
        """Keep-Alive."""
        self.send_message("SERVER", "PING")


    def send_camera_update(self, camera_name: str, person_list: list) -> None:
        """
        Sendet das VOLLE Live-Datenpaket an den Server.
        """
        payload = {
            "camera": camera_name,
            "persons": person_list
        }
        self.send_message("SERVER", "CAMERA_UPDATE", payload)

    def send_learned_body_data(self, track_id: int, stats: dict) -> None:
        """
        Sendet das Dictionary mit allen Körperdaten.
        """
        payload = {
            "id": track_id,
            "width": round(stats.get("width", 0), 2),
            "leg": round(stats.get("leg", 0), 2),
            "height": round(stats.get("height", 0), 2),
            "samples": stats.get("samples", 0)
        }

        logging.info(f"Sende Bio-Update für ID {track_id}: {payload}")
        self.send_message("SERVER", "LEARN_PERSON_STATS", payload)


    def send_db_camera_settings(self, camera_name: str, settings: dict) -> None:
        """Sendet aktualisierte Kamera-Settings direkt als Payload an den Server."""
        logging.info(f"Sende DB-Update für Kamera-Settings ({camera_name}).")
        self.send_message("SERVER", "DB_UPDATE_CAMERA_SETTINGS", settings)

    def send_db_global_room(self, width: float, height: float, depth: float) -> None:
        """Sendet geänderte Raummaße an die zentrale 'global_room' Tabelle der Datenbank."""
        payload = {
            "room_dimensions": {
                "width": float(width),
                "height": float(height),
                "depth": float(depth)
            }
        }
        logging.info(f"Sende DB-Update für globale Raummaße (W:{width}, H:{height}, D:{depth}).")
        self.send_message("SERVER", "DB_UPDATE_GLOBAL_ROOM", payload)
    def send_db_global_rectangles(self, rectangles: list) -> None:
            """Sendet die 3D-Koordinaten der Vierecke an die zentrale 'world_rectangles' Tabelle."""
            payload = {"rectangles": rectangles}
            logging.info("Sende DB-Update für 3D-Vierecke.")
            self.send_message("SERVER", "DB_UPDATE_GLOBAL_RECTANGLES", payload)

    def send_db_camera_pixels(self, camera_name: str, pixel_mapping: dict) -> None:
            """Sendet die 2D-Pixel-Mappings einer Kamera in die 'camera_pixel_mapping' Tabelle."""
            payload = {
                "camera": camera_name,
                "pixels": pixel_mapping
            }
            logging.info(f"Sende DB-Update für Pixel-Mappings ({camera_name}).")
            self.send_message("SERVER", "DB_UPDATE_CAMERA_PIXELS", payload)

    def send_db_request_full_config(self) -> None:
            """Fragt beim Start des Clients seine Config aus der PostgreSQL-Datenbank ab."""
            logging.info("Fordere Datenbank-Config vom Server an...")
            self.send_message("SERVER", "DB_REQUEST_CONFIG")
    def __loop(self) -> None:
        """Interne Schleife für Pings und Watchdog."""
        seconds_counter = 0
        while self.__is_running:
            time.sleep(1)
            seconds_counter += 1

            if self.client_connector.is_connected():
                if (hasattr(self.client_connector, 'command_handler') and
                        self.client_connector.command_handler and
                        hasattr(self.client_connector.command_handler, 'controller')):
                    self.client_connector.command_handler.controller.check_watchdog()
                if seconds_counter % self.PING_INTERVAL == 0:
                    self.send_ping()
            else:
                seconds_counter = 0

    def send_message(self, target: str, action: str, payload: Any = None) -> None:
        """
        Baut das Binär-Paket über MessagePack und setzt einen 4-Byte Längen-Header davor.
        """
        envelope = {
            "target": target,
            "action": action,
            "payload": payload,
            "sender": self.client_name,
            "timestamp": time.time()
        }

        try:
            # 1. Daten in Binärformat komprimieren
            packed_bytes = msgpack.packb(envelope, use_bin_type=True)

            # 2. Länge des Pakets berechnen
            msg_length = len(packed_bytes)

            # 3. Länge als 4-Byte Integer formatieren
            length_prefix = struct.pack('>I', msg_length)

            # 4. Header und Payload zusammenfügen und senden
            final_packet = length_prefix + packed_bytes

            self.client_connector.send_raw_packet(final_packet)

        except Exception as e:
            logging.error(f"Sendefehler (MessagePack) bei {action}: {e}")