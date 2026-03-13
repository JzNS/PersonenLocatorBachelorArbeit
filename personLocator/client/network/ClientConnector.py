import socket
import time
import logging
import struct
import msgpack
from typing import Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [CLIENT] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class ClientConnector:
    """
    Verwaltet die TCP-Verbindung zum Server.
    Empfängt MessagePack-Binärdaten mit 4-Byte Längen-Header.
    """
    RETRY_INTERVAL = 10

    def __init__(
            self,
            server_ip: str,
            server_port: int,
            client_name: str,
            command_handler: Optional[Any] = None
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_name = client_name
        self.command_handler = command_handler
        self.command_sender: Optional[Any] = None

        self.__socket: Optional[socket.socket] = None
        self.__is_connected = False

    def is_connected(self) -> bool:
        return self.__is_connected

    def start(self) -> None:
        """Startet die Verbindungs- und Empfangsschleife."""
        while True:
            try:
                self.__connect_and_run()
            except KeyboardInterrupt:
                logging.info("Client wird manuell gestoppt...")
                break
            except Exception as e:
                logging.error(f"Verbindungsfehler: {e}")

            self.__cleanup()
            logging.info(f"Warte {self.RETRY_INTERVAL} Sekunden bis zum nächsten Versuch...")
            time.sleep(self.RETRY_INTERVAL)

    def set_command_sender(self, sender: Any) -> None:
        self.command_sender = sender

    def set_command_handler(self, handler: Any) -> None:
        self.command_handler = handler

    def send_raw_packet(self, data: bytes) -> None:
        if self.__is_connected and self.__socket:
            try:
                self.__socket.sendall(data)
            except OSError as e:
                logging.error(f"Fehler beim Senden: {e}")
                self.__is_connected = False

    def force_reconnect(self) -> None:
        logging.warning("Erzwinge Reconnect aufgrund von Timeout/Fehler...")
        self.__cleanup()


    def __connect_and_run(self) -> None:
        """Stellt die Verbindung her und startet die Empfangsschleife."""
        logging.info(f"Verbinde zu {self.server_ip}:{self.server_port}...")

        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket.settimeout(10.0)
        self.__socket.connect((self.server_ip, self.server_port))
        self.__socket.settimeout(None)

        self.__is_connected = True
        logging.info(f"Verbunden als '{self.client_name}'.")

        if self.command_handler and hasattr(self.command_handler, 'controller'):
            if hasattr(self.command_handler.controller, 'on_network_connect'):
                self.command_handler.controller.on_network_connect()

        self.__listen_loop()

    def __recv_exact(self, n: int) -> Optional[bytes]:
        """Liest exakt n Bytes vom Socket."""
        if not self.__socket: return None
        data = bytearray()
        while len(data) < n:
            try:
                packet = self.__socket.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except OSError:
                return None
        return bytes(data)

    def __listen_loop(self) -> None:
        """Blockierende Schleife, die binäre Pakete liest."""
        try:
            while self.__is_connected:
                # 1. 4 Bytes Header lesen
                header = self.__recv_exact(4)
                if not header:
                    raise ConnectionResetError("Server hat die Verbindung geschlossen (EOF).")

                # 2. Länge entpacken
                msg_len = struct.unpack('>I', header)[0]

                # 3. Payload lesen
                payload_data = self.__recv_exact(msg_len)
                if not payload_data:
                    raise ConnectionResetError("Verbindung beim Lesen des Payloads abgebrochen.")

                # 4. MessagePack entpacken
                if self.command_handler:
                    try:
                        msg = msgpack.unpackb(payload_data, raw=False, strict_map_key=False)
                        self.command_handler.handle(msg)
                    except Exception as e:
                        logging.warning(f"Fehler beim Entpacken von MessagePack: {e}")

        except (OSError, ValueError, ConnectionError) as e:
            logging.error(f"Verbindung unterbrochen: {e}")
        finally:
            self.__is_connected = False

    def __cleanup(self) -> None:
        self.__is_connected = False
        if self.__socket:
            try:
                self.__socket.close()
            except OSError:
                pass
            self.__socket = None