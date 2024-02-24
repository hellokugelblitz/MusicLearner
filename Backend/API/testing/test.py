import librosa
import librosa.display
import numpy as np

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]  

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    
    # Tempo / Beats per minute
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = tempo * 60
    
    # Key
    key = librosa.estimate_tuning(y=y, sr=sr)
    key_index = round(key) % 12 
    estimated_key = KEY_NAMES[int(key_index)]
    if key < 0:
        estimated_key += " minor"
    else: 
        estimated_key += " major"
    
    # # Chroma features
    # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # # MFCC
    # mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    # # Other features to extract
    # rmse = librosa.feature.rms(y=y)
    # cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # zcr = librosa.feature.zero_crossing_rate(y)
    
    results = {
        "bpm": bpm, 
        "key": estimated_key, 
        # "chroma": chroma_stft,
        # "mfcc": mfcc,
        # "rmse": rmse,
        # "cent": cent,
        # "rolloff": rolloff, 
        # "zcr": zcr
    }
    
    return results

features = extract_features("C:\\Users\\Sahil Dayal\\Desktop\\BRICKHACK24\\MusicLearner\\test2.mp3")
print("BPM: ", features["bpm"]) # Tempo / Beats per minute
print("Key: ", features["key"]) # Estimated key of the song