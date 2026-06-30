import time
import logging
import threading
import struct
import msgpack
from typing import Any, Optional, Dict, List, Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client.network.ClientConnector import ClientConnector


class CommandSender:
    """
    Verantwortlich für das Erstellen und Versenden von Protokoll-Paketen.
    Nutzt MessagePack und Length-Prefixing (4-Byte Header) für maximale Performance.
    """
    PING_INTERVAL: int = 20

    def __init__(self, client_connector: 'ClientConnector', client_name: str) -> None:
        self.client_connector: 'ClientConnector' = client_connector
        self.client_name: str = client_name
        self.__is_running: bool = True
        self._thread: Optional[threading.Thread] = None
        self.camera_update_count: int = 0

    def start(self) -> None:
        """Startet den Hintergrund-Thread für Pings und Statistiken."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.__loop, daemon=True, name="PingThread")
        self._thread.start()
        logging.debug("Ping-Thread gestartet.")

    def stop(self) -> None:
        self.__is_running = False

    def send_db_lens_profile(self, payload: dict[str, Any]) -> None:
        """Sendet ein neu berechnetes Linsen-Profil an die globale Server-Datenbank."""
        logging.info(f"Sende DB-Update für neues Linsenprofil: '{payload.get('name')}'")
        print("Der dis coeef" ,payload.get("dist_coeffs"))
        self.send_message("SERVER", "DB_SAVE_LENS_PROFILE", payload)

    def send_register(self) -> None:
        """Initiale Anmeldung."""
        logging.info(f"Sende Registrierung für '{self.client_name}'...")
        self.send_message("SERVER", "REGISTER", {"name": self.client_name})

    def send_ping(self) -> None:
        """Keep-Alive."""
        self.send_message("SERVER", "PING")

    def send_camera_update(self, camera_name: str, person_list: list[dict[str, Any]], meta: Optional[dict[str, Any]] = None) -> None:
        """
        Sendet das VOLLE Live-Datenpaket inkl. Performance-Metadaten an den Server.
        """
        self.camera_update_count += 1

        payload: dict[str, Any] = {
            "camera": camera_name,
            "persons": person_list
        }
        if meta:
            payload.update(meta)
            
        self.send_message("SERVER", "CAMERA_UPDATE", payload)

    def send_learned_body_data(self, track_id: int, stats: dict[str, Any]) -> None:
        """
        Sendet das Dictionary mit allen Körperdaten.
        """
        payload: dict[str, Any] = {
            "id": track_id,
            "width": round(float(stats.get("width", 0.0)), 2),
            "leg": round(float(stats.get("leg", 0.0)), 2),
            "height": round(float(stats.get("height", 0.0)), 2),
            "samples": stats.get("samples", 0)
        }

        logging.info(f"Sende Bio-Update für ID {track_id}: {payload}")
        self.send_message("SERVER", "LEARN_PERSON_STATS", payload)

    def send_db_camera_settings(self, camera_name: str, settings: dict[str, Any]) -> None:
        """Sendet aktualisierte Kamera-Settings direkt als Payload an den Server."""
        logging.info(f"Sende DB-Update für Kamera-Settings ({camera_name}).")
        self.send_message("SERVER", "DB_UPDATE_CAMERA_SETTINGS", settings)

    def send_db_global_room(self, width: float, height: float, depth: float) -> None:
        """Sendet geänderte Raummaße an die zentrale 'global_room' Tabelle der Datenbank."""
        payload: dict[str, Any] = {
            "room_dimensions": {
                "width": float(width),
                "height": float(height),
                "depth": float(depth)
            }
        }
        logging.info(f"Sende DB-Update für globale Raummaße (W:{width}, H:{height}, D:{depth}).")
        self.send_message("SERVER", "DB_UPDATE_GLOBAL_ROOM", payload)

    def send_db_global_rectangles(self, rectangles: list[dict[str, Any]]) -> None:
        """Sendet die 3D-Koordinaten der Vierecke an die zentrale 'world_rectangles' Tabelle."""
        payload: dict[str, Any] = {"rectangles": rectangles}
        logging.info("Sende DB-Update für 3D-Vierecke.")
        self.send_message("SERVER", "DB_UPDATE_GLOBAL_RECTANGLES", payload)

    def send_db_camera_pixels(self, camera_name: str, pixel_mapping: dict[str, Any]) -> None:
        """Sendet die 2D-Pixel-Mappings einer Kamera in die 'camera_pixel_mapping' Tabelle."""
        payload: dict[str, Any] = {
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
        """Interne Schleife für Pings, Watchdog und STATISTIKEN."""
        seconds_counter: int = 0
        last_stat_time: float = time.time()

        while self.__is_running:
            time.sleep(1)
            seconds_counter += 1
            now: float = time.time()

            time_diff: float = now - last_stat_time
            if time_diff >= 60.0:
                fps: float = self.camera_update_count / time_diff
                logging.info(
                    f"📊 [Netzwerk Stats] Sende-Rate: {fps:.1f} FPS ({self.camera_update_count} Pakete in {time_diff:.1f}s gesendet)")

                self.camera_update_count = 0
                last_stat_time = now

            if self.client_connector.is_connected():
                if (hasattr(self.client_connector, 'command_handler') and
                        self.client_connector.command_handler and
                        hasattr(self.client_connector.command_handler, 'controller')):
                    controller: Any = getattr(self.client_connector.command_handler, 'controller')
                    if hasattr(controller, 'check_watchdog'):
                        getattr(controller, 'check_watchdog')()
                if seconds_counter % self.PING_INTERVAL == 0:
                    self.send_ping()
            else:
                seconds_counter = 0

    def send_message(self, target: str, action: str, payload: Any = None) -> None:
        """
        Baut das Binär-Paket über MessagePack und setzt einen 4-Byte Längen-Header davor.
        """
        envelope: dict[str, Any] = {
            "target": target,
            "action": action,
            "payload": payload,
            "sender": self.client_name,
            "t2": time.time()
        }

        try:
            packed_bytes: bytes = msgpack.packb(envelope, use_bin_type=True)
            msg_length: int = len(packed_bytes)
            length_prefix: bytes = struct.pack('>I', msg_length)
            final_packet: bytes = length_prefix + packed_bytes

            self.client_connector.send_raw_packet(final_packet)

        except Exception as e:
            logging.error(f"Sendefehler (MessagePack) bei {action}: {e}")