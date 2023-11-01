import muspy

midi_file_path = "bach_846.mid"
music = muspy.read_midi(midi_file_path)

# print(music.tracks[0].name)

for i in music.tracks:
    print(i.name)