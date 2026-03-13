import sys
import logging
from typing import Optional

from PyQt6.QtWidgets import QApplication
from client.utils.ConfigManager import ConfigManager
from client.logic.ClientController import ClientController
import sys
import traceback

def catch_exceptions(t, val, tb):
    """Verhindert, dass PyQt6 lautlos bei 0xC0000409 abstürzt."""
    print("EIN KRITISCHER FEHLER IST AUFGETRETEN:", file=sys.stderr)
    traceback.print_exception(t, val, tb)


sys.excepthook = catch_exceptions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [CLIENT APP] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


class ClientApplication:
    def __init__(self) -> None:
        self.qt_app = QApplication(sys.argv)
        self._controller: Optional[ClientController] = None

    def main(self) -> None:
        client_name, server_ip, server_port = ConfigManager.load_config()

        final_ip = server_ip
        if server_ip.upper() == "AUTO":
            discovered_ip = ConfigManager.find_server_ip(port=server_port)
            if discovered_ip:
                final_ip = discovered_ip
            else:
                logging.critical("Kein Server gefunden.")
                sys.exit(1)

        self._controller = ClientController(
            client_name=client_name,
            target_ip=final_ip,
            target_port=server_port
        )

        self._controller.start()

        logging.info("Client läuft. GUI-Event-Loop gestartet.")
        sys.exit(self.qt_app.exec())


if __name__ == "__main__":
    app = ClientApplication()
    app.main()