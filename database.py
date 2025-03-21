"""
Módulo para armazenar aquisições em banco de dados SQLite3.
"""

import sqlite3
import numpy as np
import config
import time


def save_acquisition(data: np.ndarray):
    conn = sqlite3.connect(config.DB_FILE)
    cursor = conn.cursor()
    timestamp = int(time.time())
    table_name = f"acquisition_{timestamp}"
    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, channel_0 REAL, channel_1 REAL, channel_2 REAL, channel_3 REAL)"
    )

    for row in data.T:
        cursor.execute(f"INSERT INTO {table_name} (channel_0, channel_1, channel_2, channel_3) VALUES (?, ?, ?, ?)", tuple(row))

    conn.commit()
    conn.close()