import librosa
import librosa.display
import numpy as np
import openai
import speech_recognition as sr
from pprint import pprint

openai.api_key = "sk-oRmUhMSgvQRRwn5CeRpTT3BlbkFJ60a5AirkKNJo97VSrKp"

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

lyrics_analysis = analyze_lyrics("test3.mp3")    
# "C:\\Users\\Sahil Dayal\\Desktop\\BRICKHACK24\\MusicLearner\\Backend\\API\\testing\\test.mp3"
pprint(lyrics_analysis)


"""
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/__init__.py", line 241, in __enter__
    self.audio_reader = wave.open(self.filename_or_fileobject, "rb")
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/wave.py", line 630, in open
    return Wave_read(f)
           ^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/wave.py", line 284, in __init__
    self.initfp(f)
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/wave.py", line 251, in initfp
    raise Error('file does not start with RIFF id')
wave.Error: file does not start with RIFF id

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/__init__.py", line 246, in __enter__
    self.audio_reader = aifc.open(self.filename_or_fileobject, "rb")
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/aifc.py", line 954, in open
    return Aifc_read(f)
           ^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/aifc.py", line 358, in __init__
    self.initfp(file_object)
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/aifc.py", line 322, in initfp
    raise Error('file does not start with FORM id')
aifc.Error: file does not start with FORM id

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/__init__.py", line 272, in __enter__
    self.audio_reader = aifc.open(aiff_file, "rb")
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/aifc.py", line 954, in open
    return Aifc_read(f)
           ^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/aifc.py", line 364, in __init__
    self.initfp(f)
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/aifc.py", line 320, in initfp
    chunk = Chunk(file)
            ^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/chunk.py", line 67, in __init__
    raise EOFError
EOFError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/bibhashthapa/Desktop/MusicLearner/Backend/API/testing/test.py", line 119, in <module>
    lyrics_analysis = analyze_lyrics("test.mp3")    
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/bibhashthapa/Desktop/MusicLearner/Backend/API/testing/test.py", line 14, in analyze_lyrics
    with sr.AudioFile(audio_file) as source:
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/__init__.py", line 274, in __enter__
    raise ValueError("Audio file could not be read as PCM WAV, AIFF/AIFF-C, or Native FLAC; check if file is corrupted or in another format")
ValueError: Audio file could not be read as PCM WAV, AIFF/AIFF-C, or Native FLAC; check if file is corrupted or in another format
bibhashthapa@bibhashs-mbp testing % python3 test.py
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 1348, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/http/client.py", line 1282, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/http/client.py", line 1328, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/http/client.py", line 1277, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/http/client.py", line 1076, in _send_output
    self.send(chunk)
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/http/client.py", line 998, in send
    self.sock.sendall(data)
BrokenPipeError: [Errno 32] Broken pipe

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/recognizers/google.py", line 205, in obtain_transcription
    response = urlopen(request, timeout=timeout)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 519, in open
    response = self._open(req, data)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 536, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 496, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 1377, in http_open
    return self.do_open(http.client.HTTPConnection, req)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 1351, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 32] Broken pipe>

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/bibhashthapa/Desktop/MusicLearner/Backend/API/testing/test.py", line 119, in <module>
    lyrics_analysis = analyze_lyrics("test3.wav")    
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/bibhashthapa/Desktop/MusicLearner/Backend/API/testing/test.py", line 16, in analyze_lyrics
    lyrics = r.recognize_google(audio)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/recognizers/google.py", line 244, in recognize_legacy
    response_text = obtain_transcription(
                    ^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/recognizers/google.py", line 209, in obtain_transcription
    raise RequestError(
speech_recognition.exceptions.RequestError: recognition connection failed: [Errno 32] Broken pipe
bibhashthapa@bibhashs-mbp testing % python3 test.py
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/recognizers/google.py", line 205, in obtain_transcription
    response = urlopen(request, timeout=timeout)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 525, in open
    response = meth(req, response)
               ^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 634, in http_response
    response = self.parent.error(
               ^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 563, in error
    return self._call_chain(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 496, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/urllib/request.py", line 643, in http_error_default
    raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 400: Bad Request

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/bibhashthapa/Desktop/MusicLearner/Backend/API/testing/test.py", line 119, in <module>
    lyrics_analysis = analyze_lyrics("test3.wav")    
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/bibhashthapa/Desktop/MusicLearner/Backend/API/testing/test.py", line 16, in analyze_lyrics
    lyrics = r.recognize_google(audio)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/recognizers/google.py", line 244, in recognize_legacy
    response_text = obtain_transcription(
                    ^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/speech_recognition/recognizers/google.py", line 207, in obtain_transcription
    raise RequestError("recognition request failed: {}".format(e.reason))
speech_recognition.exceptions.RequestError: recognition request failed: Bad Request
"""

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
