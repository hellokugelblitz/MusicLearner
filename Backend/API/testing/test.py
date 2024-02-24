import librosa
import librosa.display
import numpy as np
import openai
import speech_recognition as sr
from pprint import pprint

openai.api_key = "YOUR_OPENAI_API_KEY"

def analyze_lyrics(audio_file):
    
    # Speech recognition 
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  
    lyrics = r.recognize_google(audio)

    # Sentiment analysis
    sentiment_response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Analyze the sentiment of this text: {}\n".format(lyrics),
      temperature=0.7,
      max_tokens=60,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    sentiment = sentiment_response.choices[0].text

    # Summarization
    summary_response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Summarize this text: {}\n".format(lyrics),
      temperature=0.7,
      max_tokens=60,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    summary = summary_response.choices[0].text

    # Mood analysis
    moods = analyze_moods(lyrics) 
    themes = analyze_themes(lyrics)

    # Genre analysis
    genre_response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Suggest genres for this song based on the lyrics: {}\n".format(lyrics),
      temperature=0.5,
      max_tokens=60,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    genre = genre_response.choices[0].text    

    # Define other analysis functions    
    def analyze_moods(lyrics):
        moods_prompt = f"Analyze the moods conveyed in this song lyric text: {lyrics}\n"
        moods_prompt += "Categorize presence of these moods as percentages (0-100%):\n"
        moods_prompt += "- Happy\n- Sad\n- Angry\n- Calm\n- Romantic\n- Hopeful\n"

        moods_response = openai.Completion.create(
            engine="text-davinci-003", 
            prompt=moods_prompt,
            temperature=0.5,
            max_tokens=150,
        )
        
        moods_text = moods_response.choices[0].text
        mood_lines = moods_text.splitlines()
        
        mood_dict = {}
        for line in mood_lines:
            if ":" in line:
                mood_name, mood_percent = line.split(":")
                mood_name = mood_name.strip()
                mood_percent = int(mood_percent.strip()[:-1]) 
                mood_dict[mood_name] = mood_percent
        
        return mood_dict
    
    def analyze_themes(lyrics):
        themes_prompt = f"Analyze the main themes conveyed in this song lyric text: {lyrics}\n" 
        themes_prompt += "Identify presence of these themes as percentages (0-100%):\n"
        themes_prompt += "- Love\n- Empowerment\n- Loneliness\n- Heartbreak\n- Self-care\n- Inspiration\n"

        themes_response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=themes_prompt, 
            temperature=0.5,
            max_tokens=150,
        )
        
        themes_text = themes_response.choices[0].text
        theme_lines = themes_text.splitlines()
        
        theme_dict = {}
        for line in theme_lines:
            if ":" in line:
                theme_name, theme_percent = line.split(":")
                theme_name = theme_name.strip()
                theme_percent = int(theme_percent.strip()[:-1])
                theme_dict[theme_name] = theme_percent

        return theme_dict

    results = {
        "summary": summary,
        "sentiment": sentiment, 
        "moods": moods,
        "themes": themes,
        "genre": genre
    }
    
    return results

lyrics_analysis = analyze_lyrics("C:\\Users\\Sahil Dayal\\Desktop\\BRICKHACK24\\MusicLearner\\Backend\\API\\testing\\test.mp3")    
# "C:\\Users\\Sahil Dayal\\Desktop\\BRICKHACK24\\MusicLearner\\Backend\\API\\testing\\test.mp3"
pprint(lyrics_analysis)

# KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]  

# def extract_features(audio_file):
#     y, sr = librosa.load(audio_file)
    
#     # Tempo / Beats per minute
#     tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#     bpm = tempo * 60
    
#     # Key
#     key = librosa.estimate_tuning(y=y, sr=sr)
#     key_index = round(key) % 12 
#     estimated_key = KEY_NAMES[int(key_index)]
#     if key < 0:
#         estimated_key += " minor"
#     else: 
#         estimated_key += " major"
    
#     # # Chroma features
#     # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    
#     # # MFCC
#     # mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
#     # # Other features to extract
#     # rmse = librosa.feature.rms(y=y)
#     # cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#     # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     # zcr = librosa.feature.zero_crossing_rate(y)
    
#     results = {
#         "bpm": bpm, 
#         "key": estimated_key, 
#         # "chroma": chroma_stft,
#         # "mfcc": mfcc,
#         # "rmse": rmse,
#         # "cent": cent,
#         # "rolloff": rolloff, 
#         # "zcr": zcr
#     }
    
#     return results

# features = extract_features("C:\\Users\\Sahil Dayal\\Desktop\\BRICKHACK24\\MusicLearner\\test2.mp3")
# print("BPM: ", features["bpm"]) # Tempo / Beats per minute
# print("Key: ", features["key"]) # Estimated key of the song
