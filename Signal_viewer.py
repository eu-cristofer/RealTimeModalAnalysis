import sqlite3
import numpy as np
import plotly.graph_objects as go
import pandas as pd

class SignalReader:
    """
    Class to read and plot signals stored in the SQLite database.
    """
    def __init__(self, db_name="modal_analysis.db"):
        """
        Initializes the database connection.
        
        Parameters
        ----------
        db_name : str
            Name of the SQLite database file.
        """
        self.db_name = db_name
    
    def fetch_signals(self):
        """
        Fetches all signals from the database.
        
        Returns
        -------
        signals : list of np.ndarray
            A list of signal arrays retrieved from the database.
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT signal FROM signals")
        records = cursor.fetchall()
        conn.close()
        
        signals = [np.frombuffer(record[0], dtype=np.float64) for record in records]
        return signals
    
    def plot_signals(self, signals):
        """
        Plots the signals using Plotly.
        
        Parameters
        ----------
        signals : list of np.ndarray
            A list of signal arrays to be plotted.
        """
        if not signals:
            print("No signals found in the database.")
            return
        
        time_series = np.arange(len(signals[0]))
        
        fig = go.Figure()
        for idx, signal in enumerate(signals):
            fig.add_trace(go.Scatter(x=time_series, y=signal, mode='lines', name=f'Signal {idx+1}'))
        
        fig.update_layout(title="Stored Signals", xaxis_title="Time", yaxis_title="Amplitude")
        fig.show()
    
if __name__ == "__main__":
    reader = SignalReader()
    signals = reader.fetch_signals()
    reader.plot_signals(signals)
