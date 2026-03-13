import json
import threading
import logging
import time
from typing import Optional

from PyQt6.QtCore import QMetaObject, Qt, pyqtSlot, QObject, pyqtSignal

from client.gui.CalibrationWindow import CalibrationWindow
from client.network.ClientConnector import ClientConnector
from client.network.logic.ClientCommandHandler import CommandHandler
from client.network.logic.ClientCommandSender import CommandSender
from client.utils.ConfigManager import ConfigManager


class ClientController(QObject):
    WATCHDOG_TIMEOUT = 60
    sig_open_calibration = pyqtSignal()

    def __init__(self, client_name: str, target_ip: str, target_port: int) -> None:
        super().__init__()
        self.client_name = client_name
        self.target_ip = target_ip
        self.target_port = target_port

        self.camera_settings = ConfigManager.load_camera_config()

        self.connector: Optional[ClientConnector] = None
        self.handler: Optional[CommandHandler] = None
        self.sender: Optional[CommandSender] = None
        self.calibration_ui: Optional[CalibrationWindow] = None

        self.__setup_network_stack()
        self._last_pong_time = time.time()
        self.sig_open_calibration.connect(self.open_calibration_tool)

    @property
    def worker(self):
        """
        Erlaubt dem CommandHandler Zugriff auf den CalibrationWorker,
        der eigentlich im CalibrationWindow liegt.
        """
        if self.calibration_ui and hasattr(self.calibration_ui, 'worker'):
            return self.calibration_ui.worker
        return None



    def __setup_network_stack(self) -> None:
        """Erstellt Instanzen und verknüpft sie via Setter Injection."""
        logging.info(f"Initialisiere Client-Stack für {self.target_ip}:{self.target_port}...")

        self.connector = ClientConnector(
            server_ip=self.target_ip,
            server_port=self.target_port,
            client_name=self.client_name
        )

        self.handler = CommandHandler(main_window=self)
        self.sender = CommandSender(
            client_connector=self.connector,
            client_name=self.client_name
        )

        self.connector.set_command_handler(self.handler)
        self.connector.set_command_sender(self.sender)

    def open_calibration_tool(self) -> None:
        """Wird durch das Signal im Main-Thread ausgeführt."""
        if self.calibration_ui:
            self.calibration_ui.raise_()
            return

        logging.info(f"Öffne Kalibrierung für {self.client_name}")

        self.calibration_ui = CalibrationWindow(self.client_name)

        if self.sender:
            self.calibration_ui.set_network_sender(self.sender)
        else:
            logging.warning("ACHTUNG: Kalibrierung gestartet ohne Netzwerk-Sender!")

        self.calibration_ui.calibration_saved.connect(self.send_local_config_to_server)
        self.calibration_ui.show()

    def send_local_config_to_server(self) -> None:
        """
        Lädt die soeben gespeicherte lokale Config und sendet sie an den Server.
        """
        try:
            logging.info("Controller: Erkanntes Speicher-Event. Sende Config an Server...")

            # 1.Config laden
            full_config = ConfigManager.load_camera_config()

            # 2. Das Dictionary direkt als Payload übergeben
            if self.sender:
                self.sender.send_message("SERVER", "PUSH_CONFIG", full_config)
                logging.info("Controller: Config-Update erfolgreich gesendet.")
            else:
                logging.warning("Controller: Kein Sender verfügbar.")

        except Exception as e:
            logging.error(f"Controller: Fehler beim Senden der Config: {e}")

    def __on_ui_closed(self) -> None:
        """Wird aufgerufen, wenn die Kalibrierungs-GUI geschlossen wird. Bereinigt Referenzen und Ressourcen."""
        self.calibration_ui = None
        logging.info("Controller: Kalibrierungs-GUI wurde geschlossen.")

    def start(self) -> None:
        """Startet alle Hintergrund-Dienste (Netzwerk & Kamera)."""
        if not self.connector or not self.sender:
            logging.error("Netzwerk-Stack nicht initialisiert. Abbruch.")
            return

        self.sender.start()
        threading.Thread(
            target=self.connector.start,
            daemon=True,
            name="ClientConnectorThread"
        ).start()

        # 2. Kalibrierungs-GUI öffnen
        self.sig_open_calibration.emit()

        logging.info("Client-Dienste erfolgreich gestartet.")

    def handle_network_message(self, message: str) -> None:
        print(f"--> [Controller] Nachricht empfangen: {message}")

    def reload_camera_settings(self) -> None:
        """Wird aufgerufen, wenn der Server eine neue Konfiguration schickt.
        Lädt die neuen Einstellungen und signalisiert der UI, dass sie sich aktualisieren soll."""
        logging.info("Controller: Neue Konfiguration erkannt. Signalisiere UI...")

        self.camera_settings = ConfigManager.get_camera_settings(self.client_name)

        if self.calibration_ui:
            QMetaObject.invokeMethod(
                self.calibration_ui,
                "sync_ui_from_cache",
                Qt.ConnectionType.QueuedConnection
            )

    def reset_watchdog(self) -> None:
        """Wird aufgerufen, wenn ein 'PONG'-Signal vom Server empfangen wird.
        Aktualisiert den Zeitstempel für die Watchdog-Überwachung."""
        self._last_pong_time = time.time()

    def check_watchdog(self) -> None:
        """Überprüft regelmäßig, ob der Server noch antwortet.
        Wenn seit dem letzten 'PONG' zu viel Zeit vergangen ist, wird ein Reconnect ausgelöst."""
        if self.connector and self.connector.is_connected():
            elapsed = time.time() - self._last_pong_time
            if elapsed > self.WATCHDOG_TIMEOUT:
                logging.error(f"Watchdog Alarm! Keine Antwort seit {int(elapsed)}s.")
                self.connector.force_reconnect()
                self._last_pong_time = time.time()

    def on_network_connect(self) -> None:
        """Callback, wenn die Verbindung zum Server erfolgreich hergestellt wurde.
        Sendet die Registrierung und aktualisiert den Status."""
        logging.info("Netzwerk-Status: Verbunden.")
        self._last_pong_time = time.time()

        if self.sender:
            from client.utils.ConfigManager import ConfigManager
            ConfigManager.set_network_sender(self.sender)
            self.sender.send_register()

    def reconnect_network(self) -> None:
        if self.connector:
            logging.info("Versuche Reconnect...")
            self.connector.force_reconnect()
            self._last_pong_time = time.time()