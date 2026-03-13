import logging
import json
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QListWidget, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor


class DatabaseViewerWindow(QDialog):
    """
    Ein professioneller PostgreSQL Inspector mit modernem Dark-Theme,
    interaktiver Sidebar, Live-Editing und Auto-Sync.
    """

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._is_loading = False

        self.setWindowTitle("🗄️ PostgreSQL Database Inspector (Live-Edit Mode)")
        self.resize(1300, 750)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)

        self._setup_ui()
        self.load_table_names()

    def _setup_ui(self) -> None:
        """Definiert das Layout und die Styles für die Datenbank-Ansicht."""
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; }
            QLabel { color: #d4d4d4; font-weight: bold; font-size: 14px; }
            QListWidget {
                background-color: #252536; color: #a6accd;
                border: 1px solid #3e3e5e; border-radius: 6px;
                font-size: 14px; padding: 5px; outline: none;
            }
            QListWidget::item { padding: 12px; border-radius: 4px; margin-bottom: 2px; }
            QListWidget::item:hover:!selected { background-color: #2f2f44; color: #ffffff; }
            QListWidget::item:selected { background-color: #007acc; color: #ffffff; font-weight: bold; }
            QTableWidget {
                background-color: #252536; color: #e0e0e0;
                gridline-color: #3e3e5e; border: 1px solid #4a4a6a;
                border-radius: 6px; font-size: 13px; font-family: monospace;
            }
            QHeaderView::section {
                background-color: #32324e; color: #ffffff; padding: 10px;
                border: none; border-right: 1px solid #252536;
                border-bottom: 2px solid #007acc; font-weight: bold; font-size: 13px;
            }
            QTableWidget::item:selected { background-color: #3d59a1; color: white; }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 1. LINKE SEITE
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(10)

        self.list_tables = QListWidget()
        self.list_tables.setFixedWidth(280)
        self.list_tables.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_tables.currentTextChanged.connect(self.load_table_data)

        sidebar_layout.addWidget(self.list_tables)
        main_layout.addLayout(sidebar_layout)

        # 2. RECHTE SEITE (Editierbare Tabelle)
        content_layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setWordWrap(True)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        self.table_widget.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.EditKeyPressed)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.itemChanged.connect(self._on_cell_edited)

        content_layout.addWidget(self.table_widget)
        main_layout.addLayout(content_layout, stretch=1)

    def _get_db_pool(self):
        """Versucht, den DB-Pool aus dem Controller zu holen. Gibt None zurück, wenn nicht verfügbar."""
        if hasattr(self.controller, 'system_db') and self.controller.system_db:
            return self.controller.system_db.db_pool
        return None

    def load_table_names(self) -> None:
        """Lädt die Namen aller Tabellen aus der PostgreSQL-Datenbank und füllt die Sidebar.
        Fügt eine Master-Übersicht hinzu."""
        pool = self._get_db_pool()
        if not pool: return

        conn = pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
                tables = [row[0] for row in cursor.fetchall()]

                tables.insert(0, "🌟 MASTER ÜBERSICHT")

                self.list_tables.blockSignals(True)
                self.list_tables.clear()
                self.list_tables.addItems(tables)
                if self.list_tables.count() > 0:
                    self.list_tables.setCurrentRow(0)
                self.list_tables.blockSignals(False)

                self.load_table_data(self.list_tables.currentItem().text())
        finally:
            pool.putconn(conn)

    @pyqtSlot()
    def external_data_updated(self):
        """Wird vom Controller aufgerufen, wenn neue API-Daten eintreffen."""
        if self.isVisible():
            current_table = self.list_tables.currentItem().text() if self.list_tables.currentItem() else None
            self.load_table_data(current_table)

    def load_table_data(self, table_name: Optional[str] = None) -> None:
        """Lädt die Daten der ausgewählten Tabelle und zeigt sie im TableWidget an.
        Bei der Master-Übersicht werden spezielle Daten angezeigt."""
        if not table_name:
            if not self.list_tables.currentItem(): return
            table_name = self.list_tables.currentItem().text()

        pool = self._get_db_pool()
        if not pool: return

        self._is_loading = True
        conn = pool.getconn()

        try:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if table_name == "🌟 MASTER ÜBERSICHT":
                    cursor.execute("""
                        SELECT c.name as "Kamera", c.performance_mode as "Performance Modus", 
                               c.resolution_width || 'x' || c.resolution_height as "Auflösung", 
                               c.target_fps as "FPS", c.render_capacity || '%' as "KI Last", 
                               c.view_options as "UI Settings"
                        FROM cameras c ORDER BY c.name;
                    """)
                else:
                    import psycopg2
                    cursor.execute(f"SELECT * FROM {psycopg2.extensions.quote_ident(table_name, conn)} LIMIT 500;")

                rows = cursor.fetchall()
                self.table_widget.clear()

                if not rows:
                    self.table_widget.setColumnCount(0)
                    self.table_widget.setRowCount(0)
                    self._is_loading = False
                    return

                columns = list(rows[0].keys())
                self.table_widget.setColumnCount(len(columns))
                self.table_widget.setHorizontalHeaderLabels(columns)
                self.table_widget.setRowCount(len(rows))

                pk_col = None
                if "id" in columns:
                    pk_col = "id"
                elif "name" in columns:
                    pk_col = "name"

                for row_idx, row_data in enumerate(rows):
                    pk_val = row_data[pk_col] if pk_col else None

                    for col_idx, col_name in enumerate(columns):
                        val = row_data[col_name]

                        if isinstance(val, (dict, list)):
                            display_text = json.dumps(val, indent=2)
                        else:
                            display_text = str(val) if val is not None else "NULL"
                            if display_text.startswith('{') or display_text.startswith('['):
                                try:
                                    parsed = json.loads(display_text)
                                    display_text = json.dumps(parsed, indent=2)
                                except Exception:
                                    pass

                        item = QTableWidgetItem(display_text)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

                        if table_name == "🌟 MASTER ÜBERSICHT" or col_name == pk_col or not pk_col:
                            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                            item.setBackground(QColor("#1e1e2e"))
                        else:
                            item.setData(Qt.ItemDataRole.UserRole, {
                                "table": table_name,
                                "col_name": col_name,
                                "pk_col": pk_col,
                                "pk_val": pk_val,
                                "original_text": display_text
                            })

                        self.table_widget.setItem(row_idx, col_idx, item)

                self.table_widget.resizeColumnsToContents()
                self.table_widget.resizeRowsToContents()

        except Exception as e:
            logging.error(f"DB Viewer Fehler (Daten laden für {table_name}): {e}")
            QMessageBox.warning(self, "Lese-Fehler", str(e))
        finally:
            pool.putconn(conn)
            self._is_loading = False

    @pyqtSlot(QTableWidgetItem)
    def _on_cell_edited(self, item: QTableWidgetItem):
        """Wird ausgelöst, wenn du eine Zelle bearbeitest und Enter drückst."""
        if self._is_loading: return

        data = item.data(Qt.ItemDataRole.UserRole)
        if not data: return

        new_text = item.text().strip()
        if new_text == data["original_text"]: return

        parsed_val = new_text
        if new_text.startswith('{') or new_text.startswith('['):
            try:
                parsed_val = json.loads(new_text)
                parsed_val = json.dumps(parsed_val)
            except json.JSONDecodeError as e:
                QMessageBox.warning(self, "JSON Fehler", f"Ungültiges JSON Format. Änderung verworfen:\n{e}")
                self._revert_cell(item, data["original_text"])
                return

        pool = self._get_db_pool()
        if not pool: return

        conn = pool.getconn()
        try:
            import psycopg2
            with conn:
                with conn.cursor() as cursor:
                    query = f"UPDATE {psycopg2.extensions.quote_ident(data['table'], conn)} " \
                            f"SET {psycopg2.extensions.quote_ident(data['col_name'], conn)} = %s " \
                            f"WHERE {psycopg2.extensions.quote_ident(data['pk_col'], conn)} = %s"
                    cursor.execute(query, (parsed_val, data["pk_val"]))

            data["original_text"] = new_text
            item.setData(Qt.ItemDataRole.UserRole, data)
            logging.info(f"✏️ DB Viewer: {data['table']}.{data['col_name']} aktualisiert.")

            # Dem Controller Bescheid geben, dass er seinen Cache neu laden soll
            if hasattr(self.controller, 'refresh_config_cache'):
                self.controller.refresh_config_cache()
            if hasattr(self.controller, 'sync_all_clients'):
                self.controller.sync_all_clients()

        except Exception as e:
            logging.error(f"SQL Update Fehler: {e}")
            QMessageBox.critical(self, "Datenbank Fehler", f"Konnte Wert nicht speichern:\n{e}")
            self._revert_cell(item, data["original_text"])
        finally:
            pool.putconn(conn)

    def _revert_cell(self, item: QTableWidgetItem, original_text: str):
        """Setzt die Zelle lautlos zurück, wenn ein Fehler auftritt."""
        self._is_loading = True
        item.setText(original_text)
        self._is_loading = False