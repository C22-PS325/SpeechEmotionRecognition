# Import libraries
import numpy as np
import pandas as pd

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files.
import librosa
import librosa.display

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import load_model


# augmentation audio data
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# extraction audio data


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result

# Main function (get extracted audio data)


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result


# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()

result = get_features(
    '/Users/mac/Documents/Sistem Informasi/Personal/Kampus Merdeka/BANGKIT 2022/Project/SpeechEmotionRecognition/OAF_base_angry.wav')  # path predict
result = pd.DataFrame(result)

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
result = scaler.fit_transform(result)

# making our data compatible to model.
result = np.expand_dims(result, axis=2)

# Predicting audio data
audio_model = load_model(
    '/Users/mac/Documents/Sistem Informasi/Personal/Kampus Merdeka/BANGKIT 2022/Project/SpeechEmotionRecognition/models.h5')
pred_test = audio_model.predict(result)
pred_test = np.argmax(pred_test[0])
print(pred_test)
label_emotion = ['Angry', 'Fear', 'Happy',
                 'Neutral', 'Sad', 'Suprise', 'disgust']
final_pred = label_emotion[pred_test]
print(final_pred)

# pred_test = encoder.fit_transform(
#     np.array(pred_test).reshape(-1, 1)).toarray()
# y_pred = encoder.inverse_transform(pred_test)

# print(y_pred)
