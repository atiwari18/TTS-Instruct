import numpy as np
import librosa
import soundfile as sf

SR = 24_000

def load_audio(path):
    #sr=SR forces the resampling to SR
    audio, _ = librosa.load(path, sr=SR)
    return audio

def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)

# (clip, pause_after_in_seconds)
SCRIPT = [
    ("./interlock/interlock_intro.mp3", 0.75), 
    ("./interlock/interlock_goal.mp3", 0.75),
    ("./interlock/interlock_how_it_works.mp3", 1), 
    ("./interlock/interlock_option_one.mp3", 2), 
    ("./interlock/interlock_sum_example.mp3", 1), 
    ("./interlock/interlock_option_two.mp3", 2), 
    ("./interlock/interlock_exact_match.mp3", 1.25),
    ("./interlock/interlock_difference_one.mp3", 1.25), 
    ("./interlock/interlock_difference_two_three.mp3", 1), 
    ("./interlock/interlock_then_link_v2.mp3", 1),
    ("./interlock/interlock_single_die.mp3", 0.75),
    ("./interlock/interlock_thats_it.mp3", 0.75), 
    ("./interlock/interlock_ready.mp3", 0.75)
]

pieces = []
for i, (path, pause) in enumerate(SCRIPT):
    clip = load_audio(path)
    pieces += [clip, silence(pause)]



full = np.concatenate(pieces)
sf.write("interlock_full_longer_pause.wav", full, SR)