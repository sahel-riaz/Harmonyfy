import os
import muspy
import numpy as np

music_numpy_dir = "muspy_numpy_dir"
output_midi_dir = "output3_midi"


def decode(music_numpy_dir, output_midi_dir):

    if not os.path.exists(output_midi_dir):
        os.makedirs(output_midi_dir)

    resolution = 12
    tempo = 70

    for root, _, files in os.walk(music_numpy_dir):
        for i, file in enumerate(files):
            if file.endswith(".npy"):
                npy_file = os.path.join(root, file)
                data = np.load(npy_file)

                notes = []

                for row in data:
                    if row[0] == 0:
                        continue
                    elif row[0] == 1:
                        note = muspy.Note(
                            time=row[1] * resolution + row[2],
                            duration=row[5],
                            pitch=row[3],
                            velocity=row[4],
                        )
                        notes.append(note)

                track = muspy.Track(notes=notes)

                music = muspy.Music(
                    tempos=[muspy.Tempo(time=0, qpm=tempo)],
                    tracks=[track]
                )

                output_file = os.path.join(output_midi_dir, f"{file[:-4]}.mid")
                muspy.write(output_file, music, kind="midi")
                print(
                    f"File {file} converted to MIDI")


decode(music_numpy_dir, output_midi_dir)
