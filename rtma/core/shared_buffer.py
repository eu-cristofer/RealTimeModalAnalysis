"""
shared_buffer.py

Shared memory for inter-process streaming between SG and RTMA.
"""

import numpy as np
from multiprocessing import shared_memory

# Match the DAQ spec
SAMPLE_RATE = 1000
CHANNELS = 4
BUFFER_SIZE = 30000  # 30 seconds @ 1000Hz

SHM_NAME = "rtma_shared_buffer"

def create_shared_buffer():
    """
    Create and return shared memory buffer (4 x BUFFER_SIZE).
    """
    size = CHANNELS * BUFFER_SIZE
    shm = shared_memory.SharedMemory(create=True, size=np.float32().nbytes * size, name=SHM_NAME)
    buffer = np.ndarray((CHANNELS, BUFFER_SIZE), dtype=np.float32, buffer=shm.buf)
    buffer[:] = 0.0
    return shm, buffer

def attach_shared_buffer():
    """
    Attach to existing shared memory buffer.
    """
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    buffer = np.ndarray((CHANNELS, BUFFER_SIZE), dtype=np.float32, buffer=shm.buf)
    return shm, buffer


def create_or_attach_shared_buffer():
    try:
        # Try creating it
        shm = shared_memory.SharedMemory(create=True,
                                         size=np.float32().nbytes * CHANNELS * BUFFER_SIZE,
                                         name=SHM_NAME)
        buffer = np.ndarray((CHANNELS, BUFFER_SIZE),
                            dtype=np.float32,
                            buffer=shm.buf)
        buffer[:] = 0.0
        return shm, buffer
    except FileExistsError:
        # Fallback to attach
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        buffer = np.ndarray((CHANNELS, BUFFER_SIZE),
                            dtype=np.float32,
                            buffer=shm.buf)
        return shm, buffer
