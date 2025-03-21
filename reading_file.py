import sqlite3
from tkinter import Tk, filedialog

# Hide the root Tk window
root = Tk()
root.withdraw()

# Open file dialog for the user to select the SQLite file
file_path = filedialog.askopenfilename(
    title="Select SQLite database file",
    filetypes=[("SQLite Database Files", "*.sqlite"), ("All Files", "*.*")]
)

# Connect to the selected SQLite database
if file_path:
    conn = sqlite3.connect(file_path)
    print(f"Connected to database: {file_path}")
else:
    print("No file selected.")
