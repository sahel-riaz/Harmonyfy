import os
import muspy
import numpy as np
import pandas as pd
import encoding as encoding

def write_numpy_to_txt(numpy_dir, output_txt_dir):
    for root, _, files in os.walk(numpy_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_file = os.path.join(root, file)
                data = np.load(npy_file)
                output_txt = os.path.join(output_txt_dir, f"{file}.txt")
                np.savetxt(output_txt, data, fmt='%d', delimiter='\t')

def main():
    maestro_dir = "maestro-v2.0.0"
    music_numpy_dir = "muspy_numpy_dir"

    if not os.path.exists(music_numpy_dir):
        os.makedirs(music_numpy_dir)
    res = []
    for root, _, files in os.walk(maestro_dir):
        i = 0
        for file in files:
            if file.endswith(".midi"):
                i += 1
                if os.path.exists(os.path.join(music_numpy_dir, f"{file}.npy")):
                    continue
                
                midi_file = os.path.join(root, file)
                music = muspy.read(midi_file)
                flag = 0
                max_beat, max_dur, total = encoding.constraint_checker(music)
                res.append([max_beat, max_dur, total])
                if flag: continue

                numpy_data = encoding.convert_to_numpy(music)
                np.save(os.path.join(music_numpy_dir, f"{file}.npy"), numpy_data)
                print('{}---------------------- NumPy DONE ---------------------'.format(i))
    df = pd.DataFrame(res, columns=['max_beat', 'max_dur', 'total'])
    df.to_csv('statistics.csv', index=False)

if __name__ == "__main__":
    main()