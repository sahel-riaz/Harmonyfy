import os
import muspy
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import encoding1 as encoding

def main():

    maestro_dir = "maestro-v2.0.0"
    train_path = "train_dataset"
    validation_path = "validation"

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    for root, _, files in os.walk(maestro_dir):
        for i, file in enumerate(files):
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = muspy.read(midi_file)
                
                numpy_data = encoding.convert_to_numpy(music)
                if numpy_data is None:
                    continue
                else:
                    np.save(os.path.join(train_path, f"{file}.npy"), numpy_data)
                    print('----------------------', i, 'NumPy DONE ---------------------')
    
    # split into train and validation
    file_paths = [file for file in os.listdir(train_path) if file.endswith(".npy")]

    _, val_files = train_test_split(file_paths, test_size=0.1, random_state=42)

    move_files(val_files, train_path, validation_path)

if __name__ == "__main__":
    main()