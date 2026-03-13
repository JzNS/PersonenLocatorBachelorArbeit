import logging
from typing import Any, Dict

from client.utils.ConfigManager import ConfigManager


class CommandHandler:
    """
    Verarbeitet eingehende Nachrichten vom Server.
    Liegt in client.network.logic.
    """

    def __init__(self, main_window: Any) -> None:
        self.controller = main_window

    def handle(self, message: Dict[str, Any]) -> None:
        """Zentrale Verteilerstelle für eingehende Befehle."""
        action = message.get("action")
        payload = message.get("payload", {})
        sender = message.get("sender", "SERVER")

        #Datenbank Server Sync
        if action in ["CONFIG_FULL", "CONFIG_LITE"]:
            self.__handle_db_config(action, payload)

        elif action == "PONG":
            self.__handle_pong()

        elif action == "PRINT":
            self.__handle_print(sender, payload)

        elif action == "UPDATE_PERSON_STATS":
            self.__handle_person_stats(payload)


        elif not action:
            logging.warning("Nachricht ohne 'action' erhalten.")
        else:
            logging.info(f"Unbekannte Action empfangen: {action}")

    def __handle_db_config(self, action: str, payload: Dict[str, Any]) -> None:
        """
        Wird direkt beim Handshake aufgerufen oder wenn der Client explizit fragt.
        Lädt die frischen PostgreSQL-Daten in den Arbeitsspeicher.
        """
        logging.info(f"📥 Lade frische Einstellungen aus der Datenbank ({action})...")

        try:
            ConfigManager.set_cached_config(payload)

            if hasattr(self.controller, 'calibration_ui') and self.controller.calibration_ui:
                if hasattr(self.controller.calibration_ui, 'worker'):
                    self.controller.calibration_ui.worker.trigger_config_reload()
                    logging.info("🔄 Worker erfolgreich mit frischen DB-Daten neu geladen!")

            if hasattr(self.controller, "reload_camera_settings"):
                self.controller.reload_camera_settings()

        except Exception as e:
            logging.error(f"Konnte Worker nach DB-Sync nicht neuladen: {e}")


    def __handle_pong(self) -> None:
        """Reagiert auf den PING des Servers. Resettet den Watchdog-Timer, damit der Server weiß, dass der Client noch lebt."""
        if hasattr(self.controller, 'reset_watchdog'):
            self.controller.reset_watchdog()

    def __handle_print(self, sender: str, payload: Any) -> None:
        """Einfacher Debug-Befehl, um Textnachrichten vom Server zu loggen. Kann für Fehlermeldungen oder Status-Updates genutzt werden."""
        msg_text = str(payload)
        logging.info(f"MSG von {sender}: {msg_text}")
        if hasattr(self.controller, "handle_network_message"):
            self.controller.handle_network_message(f"{sender}: {msg_text}")



    def __handle_person_stats(self, payload: Dict[str, Any]) -> None:
        """Verarbeitet Updates zu Personen-Statistiken, die vom Server gesendet werden.
        Erwartet eine ID und optionale Werte für Breite, Beinlänge und Höhe.
        Aktualisiert die entsprechenden Daten im PersonManager der Toolbox."""
        try:
            p_id = int(payload.get("id", -1))
            if p_id == -1: return

            stats_update = {}
            if "width" in payload: stats_update["width"] = float(payload["width"])
            if "leg" in payload: stats_update["leg"] = float(payload["leg"])
            if "height" in payload: stats_update["height"] = float(payload["height"])

            if stats_update:
                logging.info(f"Server-Update für ID {p_id}: {stats_update}")
                if (hasattr(self.controller, 'worker') and
                        self.controller.worker and
                        hasattr(self.controller.worker, 'toolbox')):
                    manager = self.controller.worker.toolbox.person_manager
                    manager.set_server_stats(p_id, stats_update)

        except Exception as e:
            logging.error(f"Fehler bei Person-Stats Update: {e}")