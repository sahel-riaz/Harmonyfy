import os
import muspy
import numpy as np

import muspy_encoding_data as encoding

def write_numpy_to_txt(numpy_dir, output_txt_dir):
    for root, _, files in os.walk(numpy_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_file = os.path.join(root, file)
                data = np.load(npy_file)
                output_txt = os.path.join(output_txt_dir, f"{file}.txt")
                np.savetxt(output_txt, data, fmt='%d', delimiter='\t')

def main():
    maestro_dir = "maestro-v3.0.0"
    music_json_dir = "muspy_files_json"
    music_notes_dir = "muspy_notes"
    music_numpy_dir = "muspy_numpy_dir"
    music_numpy_txt = 'muspy_numpy_txt'
    music_csv_dir = "muspy_csv_dir"

    if not os.path.exists(music_numpy_dir):
        os.makedirs(music_numpy_dir)

    if not os.path.exists(music_numpy_txt):
        os.makedirs(music_numpy_txt)

    for root, _, files in os.walk(maestro_dir):
        for file in files:
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = muspy.read(midi_file)
                # encoding.convert_to_json(music, file, music_json_dir)
                encoding.convert_to_csv(music, file, music_csv_dir)
                print('---------------------- CSV DONE ---------------------')
                numpy_data = encoding.convert_to_numpy(music)
                np.save(os.path.join(music_numpy_dir,
                        f"{file}.npy"), numpy_data)
                print('---------------------- NumPy DONE ---------------------')

    write_numpy_to_txt(music_numpy_dir, music_numpy_txt)


if __name__ == "__main__":
    main()
