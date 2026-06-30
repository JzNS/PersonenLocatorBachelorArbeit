import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QListWidget, QAbstractItemView, QMenu, QLabel, QWidget
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QColor, QAction, QFont

from server.core.logger import get_logger

logger = get_logger("server.gui.database_viewer")


class DatabaseViewerWindow(QDialog):
    """
    Premium PostgreSQL Inspector mit Modern UI.
    UnterstÃ¼tzt virtuelle Tabellen (Settings) und Multi-Row Deletion fÃ¼r Standard-Tabellen.
    """

    VIRTUAL_TABLES = {
        "âš™ï¸ SERVER CONFIG": "server_settings",
        "ðŸŽ¯ TRACKING CONFIG": "tracking_settings",
        "ðŸŒŸ MASTER CONFIG": "master_settings"
    }

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._is_loading = False

        self.setWindowTitle("ðŸ—„ï¸ PostgreSQL Data Studio")
        self.resize(1400, 850)

        self._setup_ui()
        self.load_table_names()

    def _setup_ui(self) -> None:
        self.setStyleSheet("""
            QDialog { background-color: #11111b; }
            QLabel#titleLabel { color: #cdd6f4; font-size: 22px; font-weight: bold; padding: 5px; }
            QListWidget {
                background-color: #1e1e2e; color: #bac2de; border: 1px solid #313244;
                border-radius: 8px; font-size: 14px; font-weight: bold; outline: none; padding: 5px;
            }
            QListWidget::item { padding: 12px; border-radius: 6px; margin-bottom: 2px; }
            QListWidget::item:hover:!selected { background-color: #313244; }
            QListWidget::item:selected { background-color: #89b4fa; color: #11111b; }
            QLineEdit {
                background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #313244;
                border-radius: 8px; padding: 10px 15px; font-size: 14px;
            }
            QLineEdit:focus { border: 1px solid #89b4fa; background-color: #25253a; }
            QTableWidget {
                background-color: #1e1e2e; color: #cdd6f4; gridline-color: #313244;
                border: 1px solid #313244; border-radius: 8px;
                font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 13px;
            }
            QTableWidget::item:selected { background-color: #45475a; color: #ffffff; }
            QHeaderView::section {
                background-color: #181825; color: #89b4fa; padding: 12px;
                font-weight: bold; border: none; border-right: 1px solid #313244;
                border-bottom: 2px solid #89b4fa; font-size: 13px;
            }
            QScrollBar:vertical { border: none; background: #11111b; width: 12px; margin: 0px; }
            QScrollBar::handle:vertical { background: #45475a; min-height: 30px; border-radius: 6px; }
            QScrollBar::handle:vertical:hover { background: #585b70; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar:horizontal { border: none; background: #11111b; height: 12px; margin: 0px; }
            QScrollBar::handle:horizontal { background: #45475a; min-width: 30px; border-radius: 6px; }
            QScrollBar::handle:horizontal:hover { background: #585b70; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
            QMenu { background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 5px; }
            QMenu::item { padding: 8px 25px 8px 20px; border-radius: 4px; }
            QMenu::item:selected { background-color: #f38ba8; color: #11111b; font-weight: bold; }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        header_layout = QHBoxLayout()
        title_label = QLabel("ðŸ—„ï¸ System Database Studio")
        title_label.setObjectName("titleLabel")
        header_layout.addWidget(title_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("ðŸ” Suchen (Live-Filter)...")
        self.search_input.textChanged.connect(self._filter_table)
        header_layout.addWidget(self.search_input, stretch=1)
        main_layout.addLayout(header_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        self.list_tables = QListWidget()
        self.list_tables.setFixedWidth(280)
        self.list_tables.currentTextChanged.connect(self.load_table_data)
        content_layout.addWidget(self.list_tables)

        self.table_widget = QTableWidget()
        self.table_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.table_widget.itemChanged.connect(self._on_cell_edited)
        self.table_widget.setWordWrap(True)
        self.table_widget.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_widget.setAlternatingRowColors(True)

        content_layout.addWidget(self.table_widget, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

    def _get_db_pool(self):
        return self.controller.system_db.db_pool if hasattr(self.controller, 'system_db') else None

    def _get_primary_key(self, cursor, table_name: str) -> Optional[str]:
        query = """
            SELECT a.attname FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary;
        """
        try:
            cursor.execute(query, (table_name,))
            row = cursor.fetchone()
            return row['attname'] if row else None
        except Exception:
            return None

    def load_table_names(self) -> None:
        pool = self._get_db_pool()
        if not pool: return
        try:
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
                    db_tables = [r[0] for r in cur.fetchall()]

                    self.list_tables.blockSignals(True)
                    self.list_tables.clear()

                    for v_name in self.VIRTUAL_TABLES.keys():
                        self.list_tables.addItem(v_name)

                    self.list_tables.addItems(db_tables)
                    self.list_tables.blockSignals(False)

                    if self.list_tables.count() > 0:
                        self.list_tables.setCurrentRow(0)
                        self.load_table_data(self.list_tables.currentItem().text())
            finally:
                pool.putconn(conn)
        except Exception as e:
            logger.error(f"DB Studio Fehler: {e}")

    def load_table_data(self, table_name: str) -> None:
        if not table_name or self._is_loading: return
        pool = self._get_db_pool()
        if not pool: return

        self._is_loading = True
        try:
            conn = pool.getconn()
            try:
                self.table_widget.blockSignals(True)
                with conn.cursor(cursor_factory=RealDictCursor) as cur:

                    if table_name in self.VIRTUAL_TABLES:
                        col = self.VIRTUAL_TABLES[table_name]
                        cur.execute(f"SELECT {col} FROM global_room WHERE id = 1")
                        row = cur.fetchone()

                        self.table_widget.clear()
                        self.table_widget.setRowCount(0)
                        self.table_widget.setColumnCount(1)
                        self.table_widget.setHorizontalHeaderLabels([f"Inhalt von {col} (JSONB)"])
                        self.table_widget.setRowCount(1)

                        data = row[col] if row and row.get(col) is not None else {}
                        if isinstance(data, str):
                            try:
                                data = json.loads(data)
                            except Exception:
                                data = {}

                        txt = json.dumps(data, indent=2)
                        item = QTableWidgetItem(txt)

                        item.setData(Qt.ItemDataRole.UserRole, {
                            "is_virtual": True,
                            "table": "global_room",
                            "col": col,
                            "orig": txt
                        })
                        self.table_widget.setItem(0, 0, item)

                    else:
                        pk_col = self._get_primary_key(cur, table_name)
                        order_by = f"ORDER BY {psycopg2.extensions.quote_ident(pk_col, conn)}" if pk_col else ""
                        cur.execute(
                            f"SELECT * FROM {psycopg2.extensions.quote_ident(table_name, conn)} {order_by} LIMIT 1000;")
                        rows = cur.fetchall()

                        self.table_widget.clear()
                        self.table_widget.setRowCount(0)
                        if not rows: return

                        cols = list(rows[0].keys())
                        self.table_widget.setColumnCount(len(cols))
                        self.table_widget.setHorizontalHeaderLabels(cols)
                        self.table_widget.setRowCount(len(rows))

                        for r_idx, row in enumerate(rows):
                            for c_idx, col in enumerate(cols):
                                val = row[col]
                                txt = json.dumps(val, indent=2) if isinstance(val, (dict, list)) else str(
                                    val if val is not None else "")
                                item = QTableWidgetItem(txt)
                                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

                                if col == pk_col:
                                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                                    item.setBackground(QColor("#181825"))
                                    font = item.font()
                                    font.setBold(True)
                                    item.setFont(font)

                                item.setData(Qt.ItemDataRole.UserRole, {
                                    "is_virtual": False,
                                    "table": table_name,
                                    "col": col,
                                    "orig": txt,
                                    "pk_col": pk_col,
                                    "pk_val": row.get(pk_col) if pk_col else (row.get('name') or row.get('id'))
                                })
                                self.table_widget.setItem(r_idx, c_idx, item)

                    self.table_widget.resizeColumnsToContents()
            finally:
                self.table_widget.blockSignals(False)
                pool.putconn(conn)
        finally:
            self._is_loading = False

    @pyqtSlot(QTableWidgetItem)
    def _on_cell_edited(self, item: QTableWidgetItem) -> None:
        if self._is_loading: return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data: return

        new_val = item.text().strip()
        if new_val == data["orig"]: return

        is_json = False
        parsed_val = new_val
        if new_val.startswith('{') or new_val.startswith('['):
            try:
                parsed_val = json.loads(new_val)
                is_json = True
            except Exception:
                QMessageBox.warning(self, "JSON Fehler", "UngÃ¼ltiges Format! Die Ã„nderung wird zurÃ¼ckgesetzt.")
                item.setText(data["orig"])
                return

        pool = self._get_db_pool()
        conn = pool.getconn()
        try:
            from psycopg2.extras import Json
            with conn:
                with conn.cursor() as cur:
                    if data.get("is_virtual"):
                        cur.execute(
                            f"UPDATE global_room SET {psycopg2.extensions.quote_ident(data['col'], conn)} = %s WHERE id = 1",
                            (Json(parsed_val),))
                    else:
                        pk_ident = psycopg2.extensions.quote_ident(data['pk_col'], conn) if data.get('pk_col') else "id"
                        val_placeholder = "%s::jsonb" if is_json else "%s"

                        query = f"UPDATE {psycopg2.extensions.quote_ident(data['table'], conn)} SET {psycopg2.extensions.quote_ident(data['col'], conn)} = {val_placeholder} WHERE {pk_ident} = %s"
                        cur.execute(query, (json.dumps(parsed_val) if is_json else parsed_val, data.get("pk_val")))

            item.setBackground(QColor("#a6e3a1"))
            QTimer.singleShot(400, lambda: item.setBackground(QColor(0, 0, 0, 0)))
            data["orig"] = new_val
            item.setData(Qt.ItemDataRole.UserRole, data)

            if hasattr(self.controller, 'refresh_config_cache'):
                self.controller.refresh_config_cache()
        except Exception as e:
            QMessageBox.critical(self, "SQL Fehler", str(e))
            item.setText(data["orig"])
        finally:
            pool.putconn(conn)

    @pyqtSlot()
    def external_data_updated(self) -> None:
        if self.isVisible():
            current = self.list_tables.currentItem()
            if current: self.load_table_data(current.text())

    def _filter_table(self, text: str) -> None:
        for row in range(self.table_widget.rowCount()):
            match = any(text.lower() in (
                self.table_widget.item(row, c).text().lower() if self.table_widget.item(row, c) else "")
                        for c in range(self.table_widget.columnCount()))
            self.table_widget.setRowHidden(row, not match)

    def _show_context_menu(self, pos) -> None:
        selected_items = self.table_widget.selectedItems()
        if not selected_items: return

        first_item = selected_items[0]
        data = first_item.data(Qt.ItemDataRole.UserRole)
        if data and data.get("is_virtual"):
            return

        selected_rows = list(set(item.row() for item in selected_items))
        count = len(selected_rows)

        menu = QMenu()
        action_text = "ðŸ—‘ï¸ Zeile löschen" if count == 1 else f"ðŸ—‘ï¸ {count} Zeilen läschen"
        del_action = QAction(action_text, self)

        del_action.triggered.connect(lambda: self._delete_multiple_rows(selected_rows))
        menu.addAction(del_action)
        menu.exec(self.table_widget.mapToGlobal(pos))

    def _delete_multiple_rows(self, rows: list) -> None:
        items_to_delete = []
        table_name = None
        pk_col = None

        for row_idx in rows:
            item = self.table_widget.item(row_idx, 0)
            data = item.data(Qt.ItemDataRole.UserRole) if item else None

            if data and not data.get("is_virtual") and data.get("pk_val") is not None:
                items_to_delete.append(data["pk_val"])
                table_name = data.get("table")
                pk_col = data.get("pk_col")

        if not items_to_delete or not table_name or not pk_col:
            QMessageBox.warning(self, "Fehler",
                                "Keine gültigen Primary Keys fÃ¼r die Löschung gefunden (evtl. fehlt der Tabelle ein Primary Key).")
            return

        msg = f"Soll der ausgewählte Eintrag gelöscht werden?" if len(
            items_to_delete) == 1 else f"Sollen die {len(items_to_delete)} ausgewählten EintrÃ¤ge wirklich gelöscht werden?"
        confirm = QMessageBox.question(self, "Löschung bestätigen", msg,
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if confirm == QMessageBox.StandardButton.Yes:
            pool = self._get_db_pool()
            conn = pool.getconn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        placeholders = ', '.join(['%s'] * len(items_to_delete))
                        query = f"DELETE FROM {psycopg2.extensions.quote_ident(table_name, conn)} WHERE {psycopg2.extensions.quote_ident(pk_col, conn)} IN ({placeholders})"
                        cur.execute(query, tuple(items_to_delete))

                self.load_table_data(table_name)
                if hasattr(self.controller, 'refresh_config_cache'):
                    self.controller.refresh_config_cache()
            except Exception as e:
                QMessageBox.critical(self, "Datenbank Fehler", f"Fehler beim Löschen:\n{str(e)}")
            finally:
                pool.putconn(conn)
