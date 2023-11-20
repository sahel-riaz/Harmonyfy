import os
import muspy
import csv
import json
import numpy as np

resolution = 12

TYPE_CODE_MAP = {
    "start-of-song": 0,
    "note": 1,
    "end-of-song": 2,
}


def adjust_resolution(music):
    """Adjust the resolution of the music."""
    music.adjust_resolution(resolution)
    for track in music:
        for note in track:
            if note.duration == 0:
                note.duration = 1
    music.remove_duplicate()


def adjust_position(position):
    position_code_map = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 9,
        9: 10,
        10: 11,
        11: 12,
        "null": 0
    }

    position = position_code_map[position]
    return position


def adjust_duration(music):
    duration_code_map = {
        0: 1,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 12,
        14: 13,
        15: 13,
        16: 14,
        17: 14,
        18: 15,
        19: 15,
        20: 16,
        21: 17,
        22: 17,
        23: 18,
        24: 18,
        25: 18,
        26: 18,
        27: 18,
        28: 19,
        29: 19,
        30: 19,
        31: 19,
        32: 19,
        33: 19,
        34: 20,
        35: 20,
        36: 20,
        37: 20,
        38: 20,
        39: 21,
        40: 21,
        41: 21,
        42: 22,
        43: 22,
        44: 22,
        45: 22,
        46: 23,
        47: 23,
        48: 23,
        49: 23,
        50: 23,
        51: 23,
        52: 23,
        53: 23,
        54: 23,
        55: 24,
        56: 24,
        57: 24,
        58: 24,
        59: 24,
        60: 24,
        61: 24,
        62: 24,
        63: 24,
        64: 24,
        65: 24,
        66: 24,
        67: 25,
        68: 25,
        69: 25,
        70: 25,
        71: 25,
        72: 25,
        73: 25,
        74: 25,
        75: 25,
        76: 25,
        77: 25,
        78: 25,
        79: 26,
        80: 26,
        81: 26,
        82: 26,
        83: 26,
        84: 26,
        85: 26,
        86: 26,
        87: 26,
        88: 26,
        89: 26,
        90: 26,
        91: 27,
        92: 27,
        93: 27,
        94: 27,
        95: 27,
        96: 27,
        97: 27,
        98: 27,
        99: 27,
        100: 27,
        101: 27,
        102: 27,
        103: 27,
        104: 27,
        105: 27,
        106: 27,
        107: 27,
        108: 27,
        109: 28,
        110: 28,
        111: 28,
        112: 28,
        113: 28,
        114: 28,
        115: 28,
        116: 28,
        117: 28,
        118: 28,
        119: 28,
        120: 28,
        121: 28,
        122: 28,
        123: 28,
        124: 28,
        125: 28,
        126: 28,
        127: 28,
        128: 28,
        129: 28,
        130: 28,
        131: 28,
        132: 28,
        133: 29,
        134: 29,
        135: 29,
        136: 29,
        137: 29,
        138: 29,
        139: 29,
        140: 29,
        141: 29,
        142: 29,
        143: 29,
        144: 29,
        145: 29,
        146: 29,
        147: 29,
        148: 29,
        149: 29,
        150: 29,
        151: 29,
        152: 29,
        153: 29,
        154: 29,
        155: 29,
        156: 29,
        157: 30,
        158: 30,
        159: 30,
        160: 30,
        161: 30,
        162: 30,
        163: 30,
        164: 30,
        165: 30,
        166: 30,
        167: 30,
        168: 30,
        169: 30,
        170: 30,
        171: 30,
        172: 30,
        173: 30,
        174: 30,
        175: 30,
        176: 30,
        177: 30,
        178: 30,
        179: 30,
        180: 30,
        181: 31,
        182: 31,
        183: 31,
        184: 31,
        185: 31,
        186: 31,
        187: 31,
        188: 31,
        189: 31,
        190: 31,
        191: 31,
        192: 31,
        193: 31,
        194: 31,
        195: 31,
        196: 31,
        197: 31,
        198: 31,
        199: 31,
        200: 31,
        201: 31,
        202: 31,
        203: 31,
        204: 31,
        205: 31,
        206: 31,
        207: 31,
        208: 31,
        209: 31,
        210: 31,
        211: 31,
        212: 31,
        213: 31,
        214: 31,
        215: 31,
        216: 31,
        217: 31,
        218: 31,
        219: 31,
        220: 31,
        221: 31,
        222: 31,
        223: 31,
        224: 31,
        225: 31,
        226: 31,
        227: 31,
        228: 31,
        229: 31,
        230: 31,
        231: 31,
        232: 31,
        233: 31,
        234: 31,
        235: 31,
        236: 31,
        237: 31,
        238: 31,
        239: 31,
        240: 31,
        241: 31,
        242: 31,
        243: 31,
        244: 31,
        245: 31,
        246: 31,
        247: 31,
        248: 31,
        249: 31,
        250: 31,
        251: 31,
        252: 31,
        253: 31,
        254: 31,
        255: 31,
        256: 31,
        257: 31,
        258: 31,
        259: 31,
        260: 31,
        261: 31,
        262: 31,
        263: 31,
        264: 31,
        265: 31,
        266: 31,
        267: 31,
        268: 31,
        269: 31,
        270: 31,
        271: 31,
        272: 31,
        273: 31,
        274: 31,
        275: 31,
        276: 31,
        277: 31,
        278: 31,
        279: 31,
        280: 31,
        281: 31,
        282: 31,
        283: 31,
        284: 31,
        285: 31,
        286: 31,
        287: 31,
        288: 31,
        289: 32,
        290: 32,
        291: 32,
        292: 32,
        293: 32,
        294: 32,
        295: 32,
        296: 32,
        297: 32,
        298: 32,
        299: 32,
        300: 32,
        301: 32,
        302: 32,
        303: 32,
        304: 32,
        305: 32,
        306: 32,
        307: 32,
        308: 32,
        309: 32,
        310: 32,
        311: 32,
        312: 32,
        313: 32,
        314: 32,
        315: 32,
        316: 32,
        317: 32,
        318: 32,
        319: 32,
        320: 32,
        321: 32,
        322: 32,
        323: 32,
        324: 32,
        325: 32,
        326: 32,
        327: 32,
        328: 32,
        329: 32,
        330: 32,
        331: 32,
        332: 32,
        333: 32,
        334: 32,
        335: 32,
        336: 32,
        337: 32,
        338: 32,
        339: 32,
        340: 32,
        341: 32,
        342: 32,
        343: 32,
        344: 32,
        345: 32,
        346: 32,
        347: 32,
        348: 32,
        349: 32,
        350: 32,
        351: 32,
        352: 32,
        353: 32,
        354: 32,
        355: 32,
        356: 32,
        357: 32,
        358: 32,
        359: 32,
        360: 32,
        361: 32,
        362: 32,
        363: 32,
        364: 32,
        365: 32,
        366: 32,
        367: 32,
        368: 32,
        369: 32,
        370: 32,
        371: 32,
        372: 32,
        373: 32,
        374: 32,
        375: 32,
        376: 32,
        377: 32,
        378: 32,
        379: 32,
        380: 32,
        381: 32,
        382: 32,
        383: 32,
        384: 32,
        "null": 0
    }

    for track in music.tracks:
        for note in track.notes:
                note.duration = duration_code_map[note.duration] 

def get_beat_and_position(music):
    res = []
    max_beat = 0
    for track in music.tracks:
        for note in track.notes:
            beat, position = divmod(note.time, resolution)
            res.append([beat, position])
            max_beat = max(max_beat, beat)
    res.append(max_beat)
    return res

def constraint_checker(music):
    flag = 0
    total = 0
    max_beat = 0
    max_dur = 0
    music.adjust_resolution(resolution)
             
    for track in music.tracks:
        total += len(track.notes)*len(music.tracks)
        for note in track.notes:
            if note.duration == 0:
                note.duration = 1
            beat, _ = divmod(note.time, resolution)
            max_beat = max(max_beat, beat)
            if note.duration > 384 or max_beat > 3078:
                flag = 1
                break
            max_dur = max(max_dur, note.duration)
        if flag == 1: break
    music.remove_duplicate()
    if (total + 2) > 16384:
        flag = 1    
    return max_beat, max_dur, total + 2


def get_section(music, total_beats):
    beginning_boundary = 0.3 * total_beats
    development_boundary = 0.6 * total_beats
    recapitulation_boundary = 0.9 * total_beats

    note_sections = []

    for track in music.tracks:
        for note in track.notes:
            beat, _ = divmod(note.time, resolution)
            if beat <= beginning_boundary:
                note_sections.append([beat, 1])
            elif beat <= development_boundary:
                note_sections.append([beat, 2])
            elif beat <= recapitulation_boundary:
                note_sections.append([beat, 3])
            else:
                note_sections.append([beat, 4])

    return note_sections


def convert_to_json(music, file, out_dir):
    print('---------------- Converting to json ----------------')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    music.save_json(
        os.path.join(out_dir, f"{file}.json"))




def convert_to_numpy(music):
    adjust_resolution(music)
    beatPos = get_beat_and_position(music)
    adjust_duration(music)
    total_beats = beatPos[-1]
    sections = get_section(music, total_beats)

    notes_array = []

    start = [TYPE_CODE_MAP['start-of-song'], 0, 0, 0, 0, 0, 0, 0]
    notes_array.append(start)

    for track in music.tracks:
        for i, note in enumerate(track.notes):
            notess = [
                TYPE_CODE_MAP['note'],
                beatPos[i][0],
                adjust_position(beatPos[i][1]),
                note.pitch,
                note.velocity,
                note.duration,
                1,
                sections[i][1]
            ]
            notes_array.append(notess)

    end = [TYPE_CODE_MAP['end-of-song'], 0, 0, 0, 0, 0, 0, 0]
    notes_array.append(end)

    result = np.array(notes_array)
    return result