import os
import muspy
import csv
import json

import muspy_encoding_data as encoding


def main():
    maestro_dir = "maestro-v3.0.0"
    music_json_dir = "muspy_files_json"
    music_notes_dir = "muspy_notes"
    music_csv_dir = "muspy_csv_dir"

    for root, _, files in os.walk(maestro_dir):
        for file in files:
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = muspy.read(midi_file)
                # encoding.convert_to_json(music, file, music_json_dir)
                encoding.convert_to_csv(music, file, music_csv_dir)
                print('---------------------- DONE ---------------------')


if __name__ == "__main__":
    main()
