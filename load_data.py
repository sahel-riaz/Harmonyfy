import os
import muspy
import numpy as np
import pandas as pd
import encoding as encoding

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
                flag, max_beat, max_dur, total = encoding.constraint_checker(music)
                print(max_beat, max_dur, total)
                # res.append([max_beat, max_dur, total])
                if flag: continue

                numpy_data = encoding.convert_to_numpy(music)
                np.save(os.path.join(music_numpy_dir, f"{file}.npy"), numpy_data)
                print('{}---------------------- NumPy DONE ---------------------'.format(i))
    # df = pd.DataFrame(res, columns=['max_beat', 'max_dur', 'total'])
    # df.to_csv('statistics.csv', index=False)

if __name__ == "__main__":
    main()