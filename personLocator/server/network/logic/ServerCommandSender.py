import logging
import time
import struct
import msgpack
from typing import Any


class ServerCommandSender:
    """
    Kümmert sich um das Erstellen und Versenden von Befehlen an Clients.
    Nutzt High-Performance MessagePack + Length Prefixing.
    """

    def __init__(self, server_connector):
        self.server_connector = server_connector


    def send_pong(self, client_name: str) -> None:
        self.__send(client_name, {"target": "CLIENT", "action": "PONG"})

    def send_kick(self, client_name: str, reason: str) -> None:
        msg = {
            "target": "CLIENT",
            "action": "KICK",
            "payload": {"reason": reason}
        }
        self.__send(client_name, msg)

    def broadcast_message(self, sender_name: str, text: str) -> None:
        msg = {
            "target": "CLIENT",
            "action": "PRINT",
            "payload": f"[{sender_name}]: {text}"
        }
        self.broadcast(msg)


    def __send(self, client_name: str, envelope: dict) -> None:
        """Private Hilfsmethode: Macht aus Dict -> Bytes (MessagePack) und sendet."""
        try:
            packed_bytes = msgpack.packb(envelope, use_bin_type=True)
            length_prefix = struct.pack('>I', len(packed_bytes))
            final_packet = length_prefix + packed_bytes

            self.server_connector.send_raw_packet(client_name, final_packet)
        except Exception as e:
            logging.error(f"Sender Fehler bei Client {client_name}: {e}")

    def send_db_global_room(self, width: float, height: float, depth: float) -> None:
        """Sendet geänderte Raummaße an die zentrale Datenbank."""
        payload = {
            "room_dimensions": {
                "width": float(width),
                "height": float(height),
                "depth": float(depth)
            }
        }
        logging.info("Sende DB-Update für globale Raummaße.")
        self.send_message("SERVER", "DB_UPDATE_GLOBAL_ROOM", payload)
    def send_message(self, client_name: str, action: str, payload: Any = None) -> None:
        """Erstellt ein Paket und sendet es an einen spezifischen Client."""
        envelope = {
            "target": "CLIENT",
            "action": action,
            "payload": payload,
            "sender": "SERVER",
            "timestamp": time.time()
        }
        self.__send(client_name, envelope)

    def broadcast(self, envelope: dict) -> None:
        """Hilfsmethode: Packt Dict und sendet an ALLE Clients gleichzeitig."""
        try:
            packed_bytes = msgpack.packb(envelope, use_bin_type=True)
            length_prefix = struct.pack('>I', len(packed_bytes))
            final_packet = length_prefix + packed_bytes

            self.server_connector.broadcast_raw_packet(final_packet)
        except Exception as e:
            logging.error(f"Broadcast Fehler: {e}")