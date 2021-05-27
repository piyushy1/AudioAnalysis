import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# set paths to the UrbanSound8K dataset and metadata file
US8K_AUDIO_PATH = os.path.abspath('UrbanSound8K/audio/')
US8K_METADATA_PATH = os.path.abspath('UrbanSound8K/metadata/UrbanSound8K.csv')
# load the csv metadata file into a Pandas DataFrame structure
us8k_metadata_df = pd.read_csv(US8K_METADATA_PATH,
                               usecols=["slice_file_name", "fold", "classID"],
                               dtype={"fold": "uint8", "classID" : "uint8"})

#print(us8k_metadata_df)

# Extract a log-mel spectrogram for each audio file in the dataset and store it into a Pandas DataFrame along with its class and fold label.

HOP_LENGTH = 512        # number of samples between successive frames
WINDOW_LENGTH = 512     # length of the window in samples
N_MEL = 128             # number of Mel bands to generate


def compute_melspectrogram_with_fixed_length(audio, sampling_rate, num_of_samples=128):
    try:
        # compute a mel-scaled spectrogram
        melspectrogram = librosa.feature.melspectrogram(y=audio,
                                                        sr=sampling_rate,
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_LENGTH,
                                                        n_mels=N_MEL)

        # convert a power spectrogram to decibel units (log-mel spectrogram)
        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

        melspectrogram_length = melspectrogram_db.shape[1]

        # pad or fix the length of spectrogram
        if melspectrogram_length != num_of_samples:
            melspectrogram_db = librosa.util.fix_length(melspectrogram_db,
                                                        size=num_of_samples,
                                                        axis=1,
                                                        constant_values=(0, -80.0))
    except Exception as e:
        print("\nError encountered while parsing files\n>>", e)
        return None

    return melspectrogram_db


SOUND_DURATION = 2.95  # fixed duration of an audio excerpt in seconds

features = []

# iterate through all dataset examples and compute log-mel spectrograms
for index, row in tqdm(us8k_metadata_df.iterrows(), total=len(us8k_metadata_df)):
    file_path = f'{US8K_AUDIO_PATH}/fold{row["fold"]}/{row["slice_file_name"]}'
    audio, sample_rate = librosa.load(file_path, duration=SOUND_DURATION, res_type='kaiser_fast')

    melspectrogram = compute_melspectrogram_with_fixed_length(audio, sample_rate)
    label = row["classID"]
    fold = row["fold"]

    features.append([melspectrogram, label, fold])

# convert into a Pandas DataFrame
us8k_df = pd.DataFrame(features, columns=["melspectrogram", "label", "fold"])

# write the Pandas DataFrame object to .pkl file
WRITE_DATA = True

if WRITE_DATA:
  us8k_df.to_pickle("us8k_df.pkl")