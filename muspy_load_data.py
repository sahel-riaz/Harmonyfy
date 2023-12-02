import os
import muspy
import numpy as np

import muspy_encoding_data as encoding


def write_numpy_to_txt(numpy_dir, output_txt_dir):
    for root, _, files in os.walk(numpy_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_file = os.path.join(root, file)
                data = np.load(npy_file, allow_pickle=True)
                # Check if data is empty or not in the expected shape
                if data.ndim > 0:  # Check if the array is not 0D
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)  # Convert 1D array to 2D
                    output_txt = os.path.join(output_txt_dir, f"{file}.txt")
                    np.savetxt(output_txt, data, fmt='%s', delimiter='\t')


def main():
    maestro_dir = "maestro-v2.0.0"
    # music_json_dir = "muspy_files_json"
    # music_notes_dir = "muspy_notes"
    music_numpy_dir = "muspy2_numpy_dir"
    music_numpy_txt = 'muspy2_numpy_txt'
    # music_csv_dir = "muspy2_csv_dir"

    if not os.path.exists(music_numpy_dir):
        os.makedirs(music_numpy_dir)

    if not os.path.exists(music_numpy_txt):
        os.makedirs(music_numpy_txt)

    for root, _, files in os.walk(maestro_dir):
        for i, file in enumerate(files):
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = muspy.read(midi_file)
                # encoding.convert_to_json(music, file, music_json_dir)
                # encoding.convert_to_csv(music, file, music_csv_dir)
                # print('---------------------- CSV DONE ---------------------')
                numpy_data = encoding.convert_to_numpy(music)
                if numpy_data is None:
                    continue
                else:
                    np.save(os.path.join(music_numpy_dir,
                            f"{file}.npy"), numpy_data)
                    print('----------------------', i,
                          'NumPy DONE ---------------------')

    write_numpy_to_txt(music_numpy_dir, music_numpy_txt)


if __name__ == "__main__":
    main()
