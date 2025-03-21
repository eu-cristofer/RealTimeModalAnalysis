"""
Configurações para aquisição e interface do sistema de análise modal experimental.
"""

SAMPLE_RATE = 1000
BUFFER_SIZE = 30000
DEVICE = "cDAQ9181-1FC6921Mod1/"
CHANNELS = [
    f"{DEVICE}ai0",
    f"{DEVICE}ai1",
    f"{DEVICE}ai2",
    f"{DEVICE}ai3"
]

SENSITIVITY = 100.0
CURRENT_EXCITATION = 0.002
DB_FILE = "modal_data.db"