import json
from pathlib import Path

class ThemeLoader:
    def __init__(self, theme_name="dark"):
        self.theme_path = Path(__file__).parent / f"{theme_name}.json"
        self.theme = self.load_theme()

    def load_theme(self):
        if self.theme_path.exists():
            with open(self.theme_path, "r") as f:
                return json.load(f)
        else:
            return {}

    def apply_theme(self, widget):
        palette = self.theme
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {palette.get("background", "#000")};
                color: {palette.get("foreground", "#FFF")};
            }}
            QDial {{
                background-color: {palette.get("control_background", "#333")};
            }}
            QPushButton {{
                font-size: 16px;
            }}
        """)
