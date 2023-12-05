import muspy

resolution = 12

midi_file_path = "/Users/sahelriaz/Documents/NITC/Sem 7/Final Year Project/preprocess-2.0/performance_test/mz_545_1.mid"
music = muspy.read(midi_file_path)

groove = muspy.metrics.groove_consistency(music, resolution)
pitch_entropy = muspy.metrics.pitch_class_entropy(music)
scale = muspy.metrics.scale_consistency(music)
pitch_in_scale_rate = muspy.metrics.pitch_in_scale_rate(music, 0, 'major')

print('groove_consistency ->', groove)
print('pitch_entropy ->', pitch_entropy)
print('scale_consistency ->', scale)
print('pitch_in_scale_rate ->', pitch_in_scale_rate)
