import json
import os

def load_theme(name):
    path = os.path.join("themes", f"{name.lower()}.json")
    with open(path, "r") as file:
        return json.load(file)

def generate_stylesheet(theme):
    return f"""
    QMainWindow {{
        background-color: {theme['background']};
    }}
    QWidget {{
        background-color: {theme['background']};
        color: {theme['foreground']};
        font-family: Segoe UI;
        font-size: 12px;
    }}
    QPushButton {{
        background-color: {theme['button']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 5px;
        min-width: 80px;
    }}
    QPushButton:hover {{
        background-color: {theme['button_hover']};
    }}
    QPushButton:pressed {{
        background-color: {theme['border']};
    }}
    QComboBox, QSpinBox, QSlider, QDoubleSpinBox {{
        background-color: {theme['button']};
        border: 1px solid {theme['border']};
        border-radius: 4px;
        padding: 3px;
    }}
    QTabWidget::pane {{
        border: 1px solid {theme['border']};
        background: {theme['button']};
    }}
    QTabBar::tab {{
        background: {theme['button']};
        border: 1px solid {theme['border']};
        padding: 8px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{
        background: {theme['button_hover']};
        border-bottom: 2px solid #81A1C1;
    }}
    QLabel {{
        color: {theme['foreground']};
    }}
    """
