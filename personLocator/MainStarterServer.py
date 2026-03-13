
import sys
import logging
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from server.logic.ServerController import ServerController
from server.gui.ServerDashboard import ServerDashboard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MAIN] - %(message)s')


# PostgreSQL Download https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Fängt alle unbehandelten Fehler ab, speichert sie und zeigt ein Popup."""
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logging.critical(f"KRITISCHER ABSTURZ:\n{error_msg}")

    with open("server_crash_log.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("SERVER ABSTURZ BERICHT:\n")
        f.write(error_msg)
        f.write("=" * 60 + "\n")

    app = QApplication.instance()
    if app:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Kritischer Fehler (Server Crash)")
        msg_box.setText("Der Server ist abgestürzt!\nDie Details wurden in 'server_crash_log.txt' gespeichert.")
        msg_box.setDetailedText(error_msg)
        msg_box.exec()

    sys.exit(1)


sys.excepthook = global_exception_handler


class ServerApplication:
    def __init__(self):
        self.app = QApplication(sys.argv)

        self.dashboard = ServerDashboard()

        self.controller = ServerController(port=65432, dashboard=self.dashboard)

    def run(self) -> None:
        self.controller.start()

        self.dashboard.show_window()
        sys.exit(self.app.exec())


if __name__ == "__main__":
    server_app = ServerApplication()
    server_app.run()