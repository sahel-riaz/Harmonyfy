import csv
import pretty_midi

def decode_csv(filename):
    data = []
    with open(filename, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Skip the header row
        for row in csv_reader:
            data.append([float(value) for value in row])
    return data



def reconstruct_midi(data, resolution):
    midi = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=0)

    for i,row in enumerate(data):
        event_type = int(row[0])
        if event_type == 1:
            continue
        elif event_type == 2:
            beat = int(row[1])
            position = (row[2])
            pitch = int(row[3])
            duration = (row[4])
            velocity = int(row[5])
            time = beat * resolution + position
            if i<=5:
                print(time,duration, time+duration)
            note = pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=time, end=time + duration
            )
            track.notes.append(note)
        elif event_type == 3:
            continue

    midi.instruments.append(track)
    return midi


def main():
    csv_filename = "MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.csv"  # Replace with your CSV file name
    resolution = 12

    data = decode_csv(csv_filename)
    reconstructed_midi = reconstruct_midi(data, resolution)

    reconstructed_midi.write("reconstructed_midi.mid")

if __name__ == "__main__":
    main()
