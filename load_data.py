import os
import pretty_midi
import csv
import numpy as np
import json

# Import the functions and variables from your original script
from encoding import encode, save_csv

def main():
    maestro_dir = "maestro-v1.0.0"
    csv_tuple_dir = "csv_tuple"
    
    if not os.path.exists(csv_tuple_dir):
        os.makedirs(csv_tuple_dir)

    for root, _, files in os.walk(maestro_dir):
        for file in files:
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = pretty_midi.PrettyMIDI(midi_file)
                encoded = encode(music)
                encoded_data = encoded.tolist()
                save_csv(os.path.join(csv_tuple_dir, f"{os.path.splitext(file)[0]}.csv"), encoded_data)

if __name__ == "__main__":
    main()
