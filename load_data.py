import os
import pretty_midi
import csv
import numpy as np
import json

from encoding import encode, save_npz

def main():
    maestro_dir = "maestro-v2.0.0"
    npy_dir = "numpy_representation"  
    
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    for root, _, files in os.walk(maestro_dir):
        for file in files:
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = pretty_midi.PrettyMIDI(midi_file)
                encoded = encode(music)
                npy_file = os.path.join(npy_dir, f"{os.path.splitext(file)[0]}.npy")
                np.save(npy_file, encoded)
if __name__ == "__main__":
    main()