import socket
import threading
import logging
from typing import Optional

DISCOVERY_MSG = b"DISCOVER_SERVER"
RESPONSE_MSG = b"IAM_SERVER"


class ServerBeacon:
    """
    Ein UDP-Service, der auf Broadcast-Anfragen von Clients lauscht
    und die Existenz des Servers bestätigt.
    """

    def __init__(self, port: int = 65432) -> None:
        self.port = port
        self._is_running = False
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Startet den Listener-Thread im Hintergrund."""
        if self._is_running:
            return

        self._is_running = True
        self._thread = threading.Thread(
            target=self.__listen_loop,
            daemon=True,
            name="ServerBeaconThread"
        )
        self._thread.start()
        logging.info(f"Server Beacon (UDP Discovery) gestartet auf Port {self.port}.")

    def stop(self) -> None:
        """Stoppt den Beacon und schließt den Socket."""
        self._is_running = False
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
        logging.info("Server Beacon gestoppt.")

    def __listen_loop(self) -> None:
        """Die interne Schleife, die auf UDP-Pakete wartet."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            self._socket.bind(('', self.port))

            while self._is_running:
                try:
                    data, addr = self._socket.recvfrom(1024)

                    if data == DISCOVERY_MSG:
                        logging.info(f"Discovery-Anfrage von {addr[0]} empfangen.")

                        self._socket.sendto(RESPONSE_MSG, addr)

                except OSError as error:
                    if self._is_running:
                        logging.error(f"Fehler im Beacon-Loop: {error}")

        except Exception as e:
            logging.critical(f"Beacon konnte nicht gestartet werden: {e}")
        finally:
            if self._socket:
                self._socket.close()