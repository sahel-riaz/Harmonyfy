import os
import muspy
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
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
    train_path = "train_section_dataset"
    validation_folder = "validation_section_dataset"
    section_name = {1: "exposition", 2: "development", 3: "recapitulation", 4: "coda"}

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    for root, _, files in os.walk(maestro_dir):
        for i, file in enumerate(files):
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = muspy.read(midi_file)
                
                encoding.convert_section_numpy(music, train_path, file) 

    for section in range(1, 5):

        section_folder = section_name[section]

        train_section = os.path.join(train_path, section_folder)

        if not os.path.exists(train_section):
            os.makedirs(train_section)


        section_files = [file for file in os.listdir(train_section) if file.endswith(".npy")]
        
        _, val_files = train_test_split(section_files, test_size=0.1, random_state=42)

        val_section_folder = os.path.join(validation_folder, f"val_{section_folder}")
        if not os.path.exists(val_section_folder):
            os.makedirs(val_section_folder)
        # print("val_files:", val_files, "train_section:", train_section, "val_section_folder:", val_section_folder)
        move_files(val_files, train_section, val_section_folder)

if __name__ == "__main__":
    main()
