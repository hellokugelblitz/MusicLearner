import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import soundfile as sf
import librosa


def load_audio_file(file_path):
    """Load audio file into waveform array
    
    Args:
        file_path (str): Path to audio file
        
    Returns:
        signal (ndarray): Audio waveform 
        sample_rate (int): Sample rate of audio file
    """
    
    # Check if file exists
    try:
        audio_file = open(file_path, 'rb')
    except FileNotFoundError:
        print(f"ERROR: File {file_path} not found!")
        return None, None
    
    # Read file using soundfile (suports more formats)
    signal, sample_rate = sf.read(file_path)
    
    # If reading fails, try librosa load (for mp3 support)
    if not signal.size: 
        signal, sample_rate = librosa.load(file_path)
        
    # Print some stats
    duration =  len(signal)/ sample_rate
    print(f'Loaded {file_path} at {sample_rate} Hz, {duration:.2f} seconds')
    
    return signal, sample_rate


def extract_features(signal, sample_rate):
    """Extract audio features from waveform signal
    
    Args:
        signal (ndarray): Audio waveform 
        sample_rate (int): Sampling rate of signal
        
    Returns:
        features (ndarray): Audio feature matrix
    """
    
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=2048, hop_length=512)
    
    # Chroma stft
    stft = librosa.stft(signal, n_fft=2048, hop_length=512)
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate) 

    # MFCC
    mfcc = librosa.feature.mfcc(signal, n_mfcc=40)
    
    # Combine into feature matrix 
    features = np.concatenate([zcr, chroma_stft, mfcc], axis=0)
    
    return features

def construct_dataframe(features, labels):
    """Construct pandas DataFrame from audio features and labels
    
    Args:
        features (ndarray): Audio feature matrix
        labels (list): List of labels
        
    Returns: 
        df (DataFrame): Features DataFrame 
    """
    
    # Validate input
    if not features.shape[0] == len(labels):
        print("Number of samples and labels don't match!")
        return
    
    # Number of samples
    num_samples = features.shape[1]
    
    # DataFrame columns/features
    feature_names = [f'feat_{i}' for i in range(features.shape[0])]
    
    # List of sample/row IDs 
    sample_id = [f'sample_{i}' for i in range(num_samples)]
        
    # Construct dataframe
    df = pd.DataFrame(data=features.T, 
                      columns=feature_names,
                      index=sample_id) 
    
    # Add labels
    df['label'] = labels
    
    return df


def split_dataset(df):
    """Split dataframe into train and test sets
    
    Args:
        df (DataFrame): Features dataframe 
        
    Returns:
        X_train, X_test, y_train, y_test splits 
    """

    # Separate features and labels
    X = df.drop('label', axis=1)  
    y = df['label']
    
    # Split dataset 80:20 train:test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train):
    """Train classification model
    
    Args:
        X_train (ndarray): Training features 
        y_train (ndarray): Training labels
        
    Returns:
        model (sklearn model): Trained model
    """

    # Create SVM classifier 
    model = SVC()
    
    # Train model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data
    
    Args:
        model (sklearn model): Trained classifier 
        X_test (ndarray): Test features
        y_test (ndarray): True test labels
        
    Returns: 
        None, print accuracy
    """
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute accuracy 
    accuracy = (y_pred == y_test).mean()
    
    # Other evaluation metrics
    precision = ...
    recall = ... 
    f1 = ...
    
    # Print metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return

# Extract features
def extract_features(audio_file):
    
    # Load audio 
    signal, sample_rate = librosa.load(audio_file)
    
    # Feature extraction
    zcr = librosa.feature.zero_crossing_rate(signal) 
    #mfcc = librosa.feature.mfcc(signal)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate)
    # Chroma FFT 

    chroma_cqt = librosa.feature.chroma_cqt(signal)
    
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(signal)
    
    features = np.concatenate([zcr, mfcc, chroma_cqt, centroid])
    
    return features

# Make predictions
def make_predictions(model, audio_file):
    
    # Extract features
    features = extract_features(audio_file)  
    
    # Model prediction
    preds = model.predict([features])
    
    return preds[0]

# # Extract features 
# def extract_features(audio_file):

#     # Load audio
#     signal, sample_rate = librosa.load(audio_file)
    
#     # Feature extraction (same as before)
#     features = // code to extract features
    
#     return features

# # Make predictions
# def make_predictions(model, audio_file):
    
#     # Extract features 
#     features = extract_features(audio_file)
    
#     # Make prediction 
#     prediction = model.predict([features])

#     return prediction[0] 

# Usage

# Workflow orchestration
  
# Load file
audio_path = "MusicLearner/test.mp3"  
features = extract_features(audio_path)
# signal, sample_rate = load_audio_file(audio_path)
  
# Extract features
# features = extract_features(signal, sample_rate)   

# Labels 
labels = ['pop', 'rock', 'hiphop', 'jazz', 'classical', 'folk', 'blues', 'metal', 'disco', 'country']

# Construct dataframe  
df = construct_dataframe(features, labels)  

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(df)  

# Train model   
model = train_model(X_train, y_train)  

# Evaluate model
evaluate_model(model, X_test, y_test)

# Prediction on new file
new_audio_file = "test.mp3"
prediction = make_predictions(model, new_audio_file)