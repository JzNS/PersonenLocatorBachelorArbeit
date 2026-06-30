import socket
import threading
import logging
from server.core.logger import get_logger
from typing import Optional, Tuple
from server.core.exceptions import NetworkError

print("DEBUG: ServerBeacon.py wird geladen...")
logger = get_logger("server.network.beacon")


DISCOVERY_MSG: bytes = b"DISCOVER_SERVER"
RESPONSE_MSG: bytes = b"IAM_SERVER"


class ServerBeacon:
    """
    Ein UDP-Service, der auf Broadcast-Anfragen von Clients lauscht
    und die Existenz des Servers bestätigt.
    """

    def __init__(self, port: int = 45432) -> None:
        self.port: int = port
        self._is_running: bool = False
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Startet den Listener-Thread im Hintergrund."""
        print(f"DEBUG: ServerBeacon.start() aufgerufen (Modul: {__name__})")
        if self._is_running:
            return

        self._is_running = True
        self._thread = threading.Thread(
            target=self.__listen_loop,
            daemon=True,
            name="ServerBeaconThread"
        )
        self._thread.start()
        logger.info(f"Server Beacon (UDP Discovery) gestartet auf Port {self.port}.")

    def stop(self) -> None:
        """Stoppt den Beacon und schließt den Socket."""
        self._is_running = False
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
        logger.info("Server Beacon gestoppt.")

    def __listen_loop(self) -> None:
        """Die interne Schleife, die auf UDP-Pakete wartet."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind auf 0.0.0.0 für bessere Windows-Kompatibilität
            self._socket.bind(('0.0.0.0', self.port))

            while self._is_running:
                try:
                    if self._socket:
                        data, addr = self._socket.recvfrom(1024)

                        if data == DISCOVERY_MSG:
                            logger.info(f"Discovery-Anfrage von {addr[0]} empfangen.")
                            self._socket.sendto(RESPONSE_MSG, addr)

                except OSError as error:
                    if self._is_running:
                        logger.error(f"Fehler im Beacon-Loop: {error}")

        except OSError as e:
            logger.debug(f"Beacon OSError: errno={e.errno}, winerror={getattr(e, 'winerror', 'N/A')}")
            if e.errno == 10013 or getattr(e, 'winerror', 0) == 10013:
                err = NetworkError(
                    "UDP Discovery Port blockiert (Zugriffsrechte)", 
                    endpoint=f"UDP {self.port}", 
                    details=f"WinError 10013: Zugriff verweigert. Ursache: Port belegt (z.B. Hyper-V Reservierung) oder fehlende Admin-Rechte. (Errno: {e.errno})"
                )
                logger.critical(str(err))
            else:
                logger.critical(f"Beacon konnte nicht gestartet werden: {e} (Errno: {e.errno})")
        except Exception as e:
            logger.critical(f"Unerwarteter Fehler im Beacon: {e}")
        finally:
            if self._socket:
                self._socket.close()
