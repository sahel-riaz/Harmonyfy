import os
import muspy
import numpy as np

music_numpy_dir = "muspy_numpy_dir"
output_midi_dir = "output3_midi"

if not os.path.exists(output_midi_dir):
    os.makedirs(output_midi_dir)

resolution = 12

for root, _, files in os.walk(music_numpy_dir):
    for i, file in enumerate(files):
        if file.endswith(".npy"):
            npy_file = os.path.join(root, file)
            data = np.load(npy_file)

            # Calculate tempo based on smaller segments
            segment_duration = 0
            segment_beats = 0
            segment_count = 0
            tempo_sum = 0

            for row in data:
                if row[0] != 0:  # Skip start and end events
                    segment_duration += row[5]  # Accumulate durations
                    segment_beats += row[4]  # Accumulate positions
                    segment_count += 1

                    # Choose a segment length (10 in this case)
                    if segment_count % 10 == 0:
                        # Calculate average duration per beat for the segment
                        duration_seconds = (60 / 60) * (4 / resolution) * \
                            (segment_duration /
                             resolution) if segment_duration != 0 else 1

                        # Calculate tempo for the segment
                        tempo = 60 / duration_seconds if duration_seconds != 0 else 1
                        tempo_sum += tempo
                        segment_duration = 0
                        segment_beats = 0

            # Calculate the average tempo across segments
            final_tempo = tempo_sum / \
                (segment_count / 10) if segment_count > 0 else 60

            notes = []  # List to hold MusPy Note objects

            # Loop through the data and convert it to MusPy Note objects
            for row in data:
                if row[0] == 0:
                    continue  # Skip the start and end events
                elif row[0] == 1:  # For note events
                    note = muspy.Note(
                        time=row[1] * resolution + row[2],  # Calculate time
                        duration=row[5],
                        pitch=row[3],
                        velocity=row[4],
                    )
                    notes.append(note)  # Append the Note object to the list

            # Create a MusPy Track and add the Notes to it
            track = muspy.Track(notes=notes)

            # Create a MusPy music object and add the track to it
            music = muspy.Music(
                tempos=[muspy.Tempo(time=0, qpm=final_tempo)],
                tracks=[track]
            )

            # Save the generated music as a MIDI file
            output_file = os.path.join(output_midi_dir, f"{file[:-4]}.mid")
            muspy.write(output_file, music, kind="midi")
            print(
                f"File {file} converted to MIDI with calculated tempo {final_tempo:.2f} BPM.")
