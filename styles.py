def apply_theme(window, theme="dark"):
    if theme == "dark":
        window.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
            }
            QWidget {
                color: #D8DEE9;
                font-family: Segoe UI;
                font-size: 12px;
            }
            QPushButton {
                background-color: #3B4252;
                border: 1px solid #4C566A;
                border-radius: 4px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #434C5E;
            }
            QComboBox {
                background-color: #3B4252;
                border: 1px solid #4C566A;
                border-radius: 4px;
                padding: 3px;
            }
        """)
    else:
        window.setStyleSheet("""
            QMainWindow {
                background-color: #F5F7FA;
            }
            QWidget {
                color: #2E3440;
                font-family: Segoe UI;
                font-size: 12px;
            }
            QPushButton {
                background-color: #E5E9F0;
                border: 1px solid #D8DEE9;
                border-radius: 4px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #D8DEE9;
            }
            QComboBox {
                background-color: #E5E9F0;
                border: 1px solid #D8DEE9;
                border-radius: 4px;
                padding: 3px;
            }
        """)
