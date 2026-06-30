import os
import sys
import logging
import argparse
import subprocess
import traceback
from typing import Optional
import ctypes
import platform
import shutil

from PyQt6.QtWidgets import QApplication
from client.utils.ConfigManager import ConfigManager
from client.logic.ClientController import ClientController


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
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", type=str, help="Überschreibt den Client-Namen", default=None)
        args, _ = parser.parse_known_args()

        client_name_data, server_ip, server_port = ConfigManager.load_config()

        if args.name is None:
            names_to_start = client_name_data if isinstance(client_name_data, list) else [client_name_data]

            if len(names_to_start) > 1:
                logging.info(f"🚀 Starte {len(names_to_start)} separate Client-Prozesse in eigenen Konsolen...")

                for name in names_to_start:
                    logging.info(f"--> Öffne neues Fenster für {name}...")
                    script_path = os.path.abspath(__file__)
                    cmd = [sys.executable, script_path, "--name", name]

                    if sys.platform == "win32":
                        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE, close_fds=True)
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", "-a", "Terminal", "-n", "--args"] + cmd, close_fds=True)
                    else:
                        if shutil.which("gnome-terminal"):
                            subprocess.Popen(["gnome-terminal", "--"] + cmd, close_fds=True)
                        elif shutil.which("konsole"):
                            subprocess.Popen(["konsole", "-e"] + cmd, close_fds=True)
                        elif shutil.which("xterm"):
                            subprocess.Popen(["xterm", "-e"] + cmd, close_fds=True)
                        else:
                            subprocess.Popen(cmd, close_fds=True)

                logging.info("Launcher beendet. Kameras laufen in separaten Fenstern weiter.")
                sys.exit(0)
            else:
                client_name = names_to_start[0]

        else:
            client_name = args.name

            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetConsoleTitleW(f"[CLIENT] PersonenLocator - {client_name}")

        logging.info(f"Starte Client-Instanz für: {client_name}")

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

        logging.info(f"Client '{client_name}' läuft. GUI-Event-Loop gestartet.")
        sys.exit(self.qt_app.exec())


if __name__ == "__main__":
    app = ClientApplication()
    app.main()