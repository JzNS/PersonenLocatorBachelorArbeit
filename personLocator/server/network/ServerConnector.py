
import socket
import threading
import time
from server.core.logger import get_logger
import struct
import msgpack
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple, Union
from server.core.exceptions import NetworkError, ConnectionError, ProtocolError

logger = get_logger("server.network.connector")

if TYPE_CHECKING:
    from server.network.ServerCommandHandler import ServerCommandHandler


class ServerConnector:
    def __init__(self, port: int = 45432) -> None:
        self.port: int = port
        self._running: bool = True
        self.handler: Optional['ServerCommandHandler'] = None
        self.clients: Dict[str, socket.socket] = {}

    def set_command_handler(self, handler: 'ServerCommandHandler') -> None:
        self.handler = handler

    def start(self) -> None:
        threading.Thread(target=self._run_tcp_listener, daemon=True, name="TCPListener").start()

    def _run_tcp_listener(self) -> None:
        """Startet den TCP-Server, der auf eingehende Verbindungen wartet und für jeden Client einen neuen Thread startet."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(('0.0.0.0', self.port))
            server_sock.listen(10)
            logger.info(f"TCP Server lauscht auf Port {self.port}...")

            while self._running:
                client_sock, addr = server_sock.accept()
                threading.Thread(
                    target=self._client_reader_loop,
                    args=(client_sock, addr),
                    daemon=True
                ).start()

    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Liest exakt n Bytes von einem Socket."""
        data: bytearray = bytearray()
        while len(data) < n:
            try:
                packet: bytes = sock.recv(n - len(data))
                if not packet:
                    if len(data) > 0:
                        raise ConnectionError("Socket", details=f"Verbindung abgebrochen während Empfang von {n} Bytes.")
                    return None
                data.extend(packet)
            except OSError as e:
                raise ConnectionError("Socket", details=f"OSError beim Empfangen: {str(e)}")
        return bytes(data)

    def _client_reader_loop(self, sock: socket.socket, addr: Tuple[str, int]) -> None:
        """Liest permanent Binärdaten von einem spezifischen Client."""
        client_name: str = f"Unknown_{addr[1]}"
        endpoint: str = f"{addr[0]}:{addr[1]}"
        
        logger.info(f"Neuer Client-Thread gestartet für {endpoint}")
        
        try:
            while self._running:
                header: Optional[bytes] = self._recv_exact(sock, 4)
                if not header:
                    logger.info(f"Client {client_name} hat die Verbindung geschlossen (EOF).")
                    break
                
                try:
                    msg_len: int = int(struct.unpack('>I', header)[0])
                except struct.error:
                    raise ProtocolError("Ungültiger Nachrichten-Header", details=f"Header: {header.hex()}")

                payload_data: Optional[bytes] = self._recv_exact(sock, msg_len)
                if not payload_data:
                    raise ConnectionError(endpoint, details="Payload konnte nicht vollständig gelesen werden.")

                try:
                    msg: dict[str, Any] = msgpack.unpackb(payload_data, raw=False, strict_map_key=False)
                    msg["t3"] = time.time()  # t3 = Server Entry, direkt nach Empfang
                except Exception as e:
                    raise ProtocolError("Nachricht konnte nicht entpackt werden (msgpack)", details=str(e))

                if "payload" in msg and isinstance(msg["payload"], dict):
                    p: dict[str, Any] = msg["payload"]
                    for key in ["name", "cam_name"]:
                        if key in p and isinstance(p[key], list):
                            val: list[Any] = p[key]
                            p[key] = str(val[0]) if len(val) > 0 else "UNKNOWN_CAM"

                actual_sender = str(msg.get("sender", client_name))

                if msg.get("action") == "REGISTER":
                    raw_name = msg.get("payload", {}).get("name", actual_sender)
                    if isinstance(raw_name, list):
                        new_name = str(raw_name[0]) if len(raw_name) > 0 else "UNKNOWN"
                    else:
                        new_name = str(raw_name)

                    if new_name != client_name:
                        if client_name in self.clients:
                            del self.clients[client_name]
                        
                        client_name = new_name
                        self.clients[client_name] = sock
                        logger.info(f"Client {endpoint} als '{client_name}' registriert.")
                    else:
                        self.clients[client_name] = sock
                    
                    actual_sender = client_name

                if self.handler:
                    self.handler.handle_message(msg, actual_sender)

        except NetworkError as ne:
            logger.error(f"Netzwerkfehler bei Client {client_name}: {ne}")
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei Client {client_name}: {e}", exc_info=True)

        finally:
            if client_name in self.clients:
                del self.clients[client_name]

            if self.handler:
                self.handler.handle_client_disconnect(client_name)

            try:
                sock.close()
            except OSError:
                pass
            logger.info(f"Verbindung zu Client {client_name} ({endpoint}) beendet.")

    def send_raw_packet(self, client_name: str, data: bytes) -> None:
        sock: Optional[socket.socket] = self.clients.get(client_name)
        if sock:
            try:
                sock.sendall(data)
            except OSError:
                logger.error(f"Senden an {client_name} fehlgeschlagen.")

    def broadcast_raw_packet(self, data: bytes) -> None:
        for name, sock in list(self.clients.items()):
            try:
                sock.sendall(data)
            except OSError:
                pass

    def get_client_ip(self, client_name: str) -> str:
        sock: Optional[socket.socket] = self.clients.get(client_name)
        if sock:
            peer: Any = sock.getpeername()
            return str(peer[0])
        return "0.0.0.0"