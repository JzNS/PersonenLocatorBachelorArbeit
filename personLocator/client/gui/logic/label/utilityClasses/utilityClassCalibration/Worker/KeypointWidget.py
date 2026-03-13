from PyQt6.QtWidgets import QVBoxLayout, QWidget, QTabWidget

from client.gui.logic.label.utilityClasses.utilityClassSideTabs.PersonDetailTab import PersonDetailTab


class KeypointWidget(QWidget):
    """
    Haupt-Widget: Verwaltet Tabs für mehrere Personen.
    """

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Tab-Container
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; background: #1a1a1a; }
            QTabBar::tab { background: #333; color: #aaa; padding: 8px; }
            QTabBar::tab:selected { background: #007acc; color: white; font-weight: bold; }
        """)
        self.layout.addWidget(self.tabs)

        self.active_tabs = {}

    def update_data(self, persons_data: list):
        """Wird vom Worker aufgerufen. Synchronisiert Tabs mit erkannten Personen."""

        if not persons_data:
            self.tabs.clear()
            self.active_tabs.clear()
            return

        current_ids = set()

        for p in persons_data:
            pid = p['id']
            current_ids.add(pid)

            if pid not in self.active_tabs:
                new_tab = PersonDetailTab()
                self.tabs.addTab(new_tab, f"Person {pid}")
                self.active_tabs[pid] = new_tab

            self.active_tabs[pid].update_view(p)

        for existing_id in list(self.active_tabs.keys()):
            if existing_id not in current_ids:
                widget_to_remove = self.active_tabs[existing_id]
                idx = self.tabs.indexOf(widget_to_remove)
                if idx != -1:
                    self.tabs.removeTab(idx)
                del self.active_tabs[existing_id]