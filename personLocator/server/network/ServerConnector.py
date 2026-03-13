# server/network/ServerConnector.py
import socket
import threading
import logging
import struct
import msgpack
from typing import Any, Dict, Optional


class ServerConnector:
    def __init__(self, port: int = 65432) -> None:
        self.port = port
        self._running = True
        self.handler = None
        self.clients: Dict[str, socket.socket] = {}

    def set_command_handler(self, handler):
        self.handler = handler

    def start(self) -> None:
        threading.Thread(target=self._run_tcp_listener, daemon=True, name="TCPListener").start()

    def _run_tcp_listener(self) -> None:
        """Startet den TCP-Server, der auf eingehende Verbindungen wartet und für jeden Client einen neuen Thread startet."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(('0.0.0.0', self.port))
            server_sock.listen(10)
            logging.info(f"TCP Server lauscht auf Port {self.port}...")

            while self._running:
                client_sock, addr = server_sock.accept()
                threading.Thread(
                    target=self._client_reader_loop,
                    args=(client_sock, addr),
                    daemon=True
                ).start()

    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Liest exakt n Bytes von einem Socket."""
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except OSError:
                return None
        return bytes(data)

    def _client_reader_loop(self, sock: socket.socket, addr: tuple) -> None:
        """Liest permanent Binärdaten von einem spezifischen Client."""
        client_name = f"Unknown_{addr[1]}"
        try:
            while self._running:
                header = self._recv_exact(sock, 4)
                if not header: break
                msg_len = struct.unpack('>I', header)[0]
                payload_data = self._recv_exact(sock, msg_len)
                if not payload_data: break
                msg = msgpack.unpackb(payload_data, raw=False, strict_map_key=False)
                if msg.get("action") == "REGISTER":
                    client_name = msg.get("payload", {}).get("name", client_name)
                    self.clients[client_name] = sock

                if self.handler:
                    self.handler.handle_message(msg, client_name)
        except Exception as e:
            logging.error(f"Fehler bei Client {client_name}: {e}")
        finally:
            if client_name in self.clients:
                del self.clients[client_name]
            if self.handler:
                self.handler.handle_client_disconnect(client_name)
            sock.close()

    def send_raw_packet(self, client_name: str, data: bytes) -> None:
        sock = self.clients.get(client_name)
        if sock:
            try:
                sock.sendall(data)
            except OSError:
                logging.error(f"Senden an {client_name} fehlgeschlagen.")

    def broadcast_raw_packet(self, data: bytes) -> None:
        for name, sock in list(self.clients.items()):
            try:
                sock.sendall(data)
            except OSError:
                pass

    def get_client_ip(self, client_name: str) -> str:
        sock = self.clients.get(client_name)
        if sock:
            return sock.getpeername()[0]
        return "0.0.0.0"