import numpy as np
import matplotlib.pyplot as plt

DIMENSIONS = ["type", "beat", "position", "pitch", "duration", "velocity", "instrument", "section"]

npy_file = "numpy_representation//MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.npy"  
data = np.load(npy_file)
print(data)
file_path = 'output.txt'

np.savetxt(file_path, data, delimiter='\t')