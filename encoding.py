import pretty_midi
import csv
import muspy
import numpy as np
import json

RESOLUTION = 12
MAX_DURATION = 384

DIMENSIONS = ["type", "beat", "position", "pitch", "duration", "velocity", "instrument", "section"]

TYPE_CODES = {"start-of-song": 1, "note": 2, "end-of-song": 3}

POSITION_CODE_MAP = {i: i + 1 for i in range(RESOLUTION)}
POSITION_CODE_MAP[None] = 0

INSTRUMENT_CODE = 1  # Piano

N_TOKENS = [len(TYPE_CODES), 0, max(POSITION_CODE_MAP.values()) + 1, 128, MAX_DURATION + 1, 128, 2]

def get_time_signature(midi):
    for event in midi.time_signature_changes:
        return event.numerator, event.denominator
    return None

def calculate_sections(music, total_duration):
    beginning_boundary = 0.3 * total_duration
    development_boundary = 0.6 * total_duration
    recapitulation_boundary = 0.9 * total_duration

    note_sections = []

    for instrument in music.instruments:
        if not instrument.is_drum and instrument.program == 0:  # Only consider piano (program 0)
            for note in instrument.notes:
                start_beat = note.start
                section = None

                if start_beat < beginning_boundary:
                    section = 1  
                elif start_beat < development_boundary:
                    section = 2  
                elif start_beat < recapitulation_boundary:
                    section = 3  
                else:
                    section = 4  

                note_sections.append(section)

    return note_sections

def extract_notes(music, resolution):
    notes = []
    for instrument in music.instruments:
        if not instrument.is_drum and instrument.program == 0:  # instrument.program == 0 is piano
            for note in instrument.notes:
                beat, position = divmod(note.start, resolution)
                notes.append((beat, position, note.pitch, note.end - note.start, note.velocity))
    notes = sorted(set(notes))
    return np.array(notes)

def encode_notes(notes, max_beat, note_sections):
    beat_dim = DIMENSIONS.index("beat")
    position_dim = DIMENSIONS.index("position")
    pitch_dim = DIMENSIONS.index("pitch")
    duration_dim = DIMENSIONS.index("duration")
    velocity_dim = DIMENSIONS.index("velocity")
    instrument_dim = DIMENSIONS.index("instrument")
    section_dim = DIMENSIONS.index("section")
    ######## can have instrument or not 
    codes = [(TYPE_CODES["start-of-song"], 0, 0, 0, 0, 0, INSTRUMENT_CODE, 0)] 

    for i, (beat, position, pitch, duration, velocity) in enumerate(notes):
        codes.append((TYPE_CODES["note"], beat, position, pitch, duration, velocity, INSTRUMENT_CODE, note_sections[i]))

    ######## can have instrument or not 
    codes.append((TYPE_CODES["end-of-song"], 0, 0, 0, 0, 0, INSTRUMENT_CODE, 0))

    return np.array(codes)

def decode_notes(codes):
    notes = []
    for row in codes:
        event_type = row[0]
        if event_type == TYPE_CODES["note"]:
            beat = row[1]
            position = row[2]
            pitch = row[3]
            duration = row[4]
            velocity = row[5]
            notes.append((beat, position, pitch, duration, velocity))
    return notes

def encode(music):
    time_signature = get_time_signature(music)
    if time_signature:
        numerator, denominator = time_signature
        beats_per_measure = numerator

        if denominator == 8:
            beats_per_measure *= 2
    else:
        beats_per_measure = 4

    max_beat = beats_per_measure * RESOLUTION

    total_duration = music.get_end_time()
    note_sections = calculate_sections(music, total_duration)
    notes = extract_notes(music, RESOLUTION)
    codes = encode_notes(notes, max_beat, note_sections)

    return codes

def decode(codes):
    notes = decode_notes(codes)
    music = reconstruct(notes, RESOLUTION)
    return music

def reconstruct(notes, resolution):
    music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(0, 100)])
    track = muspy.Track(program=0)
    for beat, position, pitch, duration, velocity in notes:
        time = beat * resolution + position
        track.notes.append(muspy.Note(time, pitch, duration, velocity=velocity))
    music.append(track)
    return music

def save_json(filename, data):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)

def save_csv(filename, data):
    with open(filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(DIMENSIONS)  
        csv_writer.writerows(data)

