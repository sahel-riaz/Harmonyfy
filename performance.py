import muspy

resolution = 12

midi_file_path = "performance_test/test_output_Db.mid"
music = muspy.read(midi_file_path)
# music.adjust_resolution(resolution)
groove = muspy.metrics.groove_consistency(music, resolution)
pitch_entropy = muspy.metrics.pitch_class_entropy(music)
scale = muspy.metrics.scale_consistency(music)
pitch_in_scale_rate = muspy.metrics.pitch_in_scale_rate(music, 1, 'major')

print('groove_consistency ->', groove)
print('pitch_entropy ->', pitch_entropy)
print('scale_consistency ->', scale)
print('pitch_in_scale_rate ->', pitch_in_scale_rate)
