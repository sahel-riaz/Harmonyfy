import muspy
import numpy as np
import csv
RESOLUTION = 12
MAX_DURATION = 384

DIMENSIONS = ["type", "beat", "position", "pitch", "duration", "velocity", "instrument", "section"]

TYPE_CODES = {"start-of-song": 1.0, "note": 2.0, "end-of-song": 3.0}

INSTRUMENT_CODE = 1  

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
        if not instrument.is_drum and instrument.program == 0: 
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
        if not instrument.is_drum and instrument.program == 0:  
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

    codes = np.zeros((len(notes) + 2, len(DIMENSIONS)), dtype=np.float64)

    codes[0, 0] = TYPE_CODES["start-of-song"]
    codes[-1, 0] = TYPE_CODES["end-of-song"]

    for i, (beat, position, pitch, duration, velocity) in enumerate(notes):
        codes[i + 1, 0] = TYPE_CODES["note"]
        codes[i + 1, beat_dim] = beat
        codes[i + 1, position_dim] = position
        codes[i + 1, pitch_dim] = pitch
        codes[i + 1, duration_dim] = duration
        codes[i + 1, velocity_dim] = velocity
        codes[i + 1, instrument_dim] = INSTRUMENT_CODE
        codes[i + 1, section_dim] = note_sections[i]
    
    return codes

def decode_notes(codes):
    mask = codes[:, 0] == TYPE_CODES["note"]
    notes = codes[mask, 1:]
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

def save_npz(filename, data):
    np.savez(filename, data=data)  

