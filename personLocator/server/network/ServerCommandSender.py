from server.core.logger import get_logger
import time
import struct
import msgpack
from typing import Any, Dict, TYPE_CHECKING, Optional

logger = get_logger("server.network.sender")


if TYPE_CHECKING:
    from server.network.ServerConnector import ServerConnector


class ServerCommandSender:
    """
    Kümmert sich um das Erstellen und Versenden von Befehlen an Clients.
    Nutzt High-Performance MessagePack + Length Prefixing.
    """

    def __init__(self, server_connector: 'ServerConnector') -> None:
        self.server_connector: 'ServerConnector' = server_connector

    def send_pong(self, client_name: str) -> None:
        self.__send(client_name, {"target": "CLIENT", "action": "PONG"})

    def send_kick(self, client_name: str, reason: str) -> None:
        msg: dict[str, Any] = {
            "target": "CLIENT",
            "action": "KICK",
            "payload": {"reason": reason}
        }
        self.__send(client_name, msg)

    def broadcast_message(self, sender_name: str, text: str) -> None:
        msg: dict[str, Any] = {
            "target": "CLIENT",
            "action": "PRINT",
            "payload": f"[{sender_name}]: {text}"
        }
        self.broadcast(msg)

    def __send(self, client_name: str, envelope: dict[str, Any]) -> None:
        """Private Hilfsmethode: Macht aus Dict -> Bytes (MessagePack) und sendet."""
        try:
            packed_bytes: bytes = msgpack.packb(envelope, use_bin_type=True)
            length_prefix: bytes = struct.pack('>I', len(packed_bytes))
            final_packet: bytes = length_prefix + packed_bytes

            self.server_connector.send_raw_packet(client_name, final_packet)
        except Exception as e:
            logger.error(f"Sender Fehler bei Client {client_name}: {e}")

    def send_db_global_room(self, width: float, height: float, depth: float) -> None:
        """Sendet geänderte Raummaße an die zentrale Datenbank."""
        payload: dict[str, Any] = {
            "room_dimensions": {
                "width": float(width),
                "height": float(height),
                "depth": float(depth)
            }
        }
        logger.info("Sende DB-Update für globale Raummaße.")
        self.send_message("SERVER", "DB_UPDATE_GLOBAL_ROOM", payload)

    def send_message(self, client_name: str, action: str, payload: Any = None) -> None:
        """Erstellt ein Paket und sendet es an einen spezifischen Client."""
        envelope: dict[str, Any] = {
            "target": "CLIENT",
            "action": action,
            "payload": payload,
            "sender": "SERVER",
            "timestamp": time.time()
        }
        self.__send(client_name, envelope)

    def broadcast(self, envelope: dict[str, Any]) -> None:
        """Hilfsmethode: Packt Dict und sendet an ALLE Clients gleichzeitig."""
        try:
            packed_bytes: bytes = msgpack.packb(envelope, use_bin_type=True)
            length_prefix: bytes = struct.pack('>I', len(packed_bytes))
            final_packet: bytes = length_prefix + packed_bytes

            self.server_connector.broadcast_raw_packet(final_packet)
        except Exception as e:
            logger.error(f"Broadcast Fehler: {e}")