import os
import muspy
from sklearn.model_selection import train_test_split
import shutil

# import encoding_part_of_piece as encoding
import encoding1 as encoding

def move_files(file_list, source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    for file in file_list:
        source_file_path = os.path.join(source_dir, file)
        destination_file_path = os.path.join(destination_dir, file)
        shutil.move(source_file_path, destination_file_path)

def main():
    maestro_dir = "maestro-v2.0.0"
    train_path = "train_dataset"
    validation_path = "validation_dataset"

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    # convert to numpy 
    for root, _, files in os.walk(maestro_dir):
        for i, file in enumerate(files):
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = muspy.read(midi_file)
                encoding.convert_part_to_numpy(music, train_path, file)

    # split into train and validation
    file_paths = [file for file in os.listdir(train_path) if file.endswith(".npy")]

    _, val_files = train_test_split(file_paths, test_size=0.1, random_state=42)

    move_files(val_files, train_path, validation_path)

if __name__ == "__main__":
    main()
