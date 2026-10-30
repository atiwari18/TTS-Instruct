import numpy as np
import librosa
import soundfile as sf

SR = 24_000
PERIOD_PAUSE = 0.75
EQUAL_PAUSE = 1.4
LONG_PAUSE = 2.5
END = 0.0

def load_audio(path):
    #sr=SR forces the resampling to SR
    audio, _ = librosa.load(path, sr=SR)
    return audio

def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)

# (clip, pause_after_in_seconds)
SCRIPT = [
    ("./tether/tts-celeste-01.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-02.mp3", PERIOD_PAUSE),
    ("./tether/tts-celeste-03.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-04.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-05.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-06.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-07.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-08.mp3", PERIOD_PAUSE),
    ("./tether/tts-celeste-09.mp3", PERIOD_PAUSE),
    ("./tether/tts-celeste-10.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-11.mp3", PERIOD_PAUSE), 

    ("./tether/tts-celeste-12.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-13.mp3", PERIOD_PAUSE),  
    ("./tether/tts-celeste-14.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-15.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-16.mp3", PERIOD_PAUSE), 
    ("./tether/tts-celeste-17.mp3", PERIOD_PAUSE),
    ("./tether/tts-celeste-18.mp3", END) 
]

pieces = []
for i, (path, pause) in enumerate(SCRIPT):
    clip = load_audio(path)
    pieces += [clip, silence(pause)]



full = np.concatenate(pieces)
sf.write("tether_3.wav", full, SR)