import os
import pretty_midi
import csv
import numpy as np
import json
import math

# Import the functions and variables from your original script
from encoding import encode, save_csv

def main():
    maestro_dir = "maestro-v2.0.0"
    csv_tuple_dir = "csv_tuple"
    target_file = 'MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.midi'
    
    if not os.path.exists(csv_tuple_dir):
        os.makedirs(csv_tuple_dir)

    for root, _, files in os.walk(maestro_dir):
        for file in files:
            if file == target_file:
                midi_file = os.path.join(root, file)
                music = pretty_midi.PrettyMIDI(midi_file)
                tempo_changes = music.get_tempo_changes()
                tempo = (tempo_changes[1][0])
                print(tempo)
                
                encoded = encode(music, tempo)
                encoded_data = encoded.tolist()
                save_csv(os.path.join(csv_tuple_dir, f"{os.path.splitext(file)[0]}.csv"), encoded_data)

if __name__ == "__main__":
    main()
