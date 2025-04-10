import json
from pathlib import Path

class ThemeManager:
    def __init__(self, theme_name="dark"):
        self.theme_name = theme_name
        self.theme = self.load_theme(theme_name)

    def load_theme(self, theme_name):
        theme_file = Path(__file__).parent / f"{theme_name}.json"
        if not theme_file.exists():
            raise FileNotFoundError(f"Theme file '{theme_file}' not found.")
        with open(theme_file, "r") as f:
            return json.load(f)

    def apply_theme(self, widget):
        """Apply theme to a given widget"""
        bg = self.theme["background_color"]
        fg = self.theme["text_color"]
        widget.setStyleSheet(f"background-color: {bg}; color: {fg};")
