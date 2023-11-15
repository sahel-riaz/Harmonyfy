import os
import muspy
import json

resolution = 12


def adjust_resolution(music):
    """Adjust the resolution of the music."""
    music.adjust_resolution(resolution)
    for track in music:
        for note in track:
            if note.duration == 0:
                note.duration = 1
    music.remove_duplicate()


def convert_to_json(music, file, out_dir):
    print('---------------- Converting to json ----------------')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # end_time = music.get_end_time()
    # if end_time > resolution * 4 * 2000 or end_time < resolution * 4 * 10:
    #     continue

    adjust_resolution(music)
    music.save_json(
        os.path.join(out_dir, f"{file}.json"))


def extract_notes_from_json(input_json, out_dir):
    with open(input_json, 'r') as file:
        data = json.load(file)
        notes = []

        if 'tracks' in data:
            for track in data['tracks']:
                if 'notes' in track:
                    notes.extend(track['notes'])

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        file_name = os.path.splitext(os.path.basename(input_json))[0]
        output_file = os.path.join(out_dir, f"{file_name}_notes.json")

        with open(output_file, 'w') as out_file:
            json.dump(notes, out_file, indent=2)


def main():
    maestro_dir = "maestro-v3.0.0"
    music_json_dir = "muspy_files_json"
    music_notes_dir = "muspy_notes"

    for root, _, files in os.walk(maestro_dir):
        for file in files:
            if file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                music = muspy.read(midi_file)
                convert_to_json(music, file, music_json_dir)
                extract_notes_from_json(os.path.join(music_json_dir, f"{file}.json"), music_notes_dir)
                print('---------------------- DONE ---------------------')



if __name__ == "__main__":
    main()
