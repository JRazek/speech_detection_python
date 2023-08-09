from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from utils import compute_spectral_density_normalized, get_audio_file_paths_in_dir, read_audio_file, compute_fft, visualize_plots, compute_mean_and_std

base_frequency = 8000

def split_audio_to_windows(audio, window_size):
    num_windows = len(audio) // window_size
    segments = []

    for i in range(num_windows):
        start_point = i * window_size
        end_point = start_point + window_size
        segment = audio[start_point:end_point]
        segments.append(segment)

    return np.asarray(segments)

def read_audio_files(files):
    audio_signals = [np.array(read_audio_file(file, base_frequency).get_array_of_samples()) for file in files]
    
    min_length = min([len(signal) for signal in audio_signals])
    
    padded_signals = [signal[0: min_length] for signal in audio_signals]
    array = np.asarray(padded_signals)

    return array


def handle_audio_file_dataset(audio_file, window_size):
    human_speech_file = np.asarray(audio_file)

    human_speech_audio_matrix = read_audio_files(human_speech_file)

    human_speech_audio_windows_matrix = np.array([split_audio_to_windows(seq, window_size) for seq in human_speech_audio_matrix])

    ffts = np.array([[compute_fft(window) for window in human_audio_sample] for human_audio_sample in human_speech_audio_windows_matrix])

    return ffts

def handle_audio_file(audio_file, window_size):
    return arr2d_to_arr1d(handle_audio_file_dataset([audio_file], window_size))


def arr2d_to_arr1d(arr2d):
    return np.reshape(arr2d, (arr2d.shape[0] * arr2d.shape[1], arr2d.shape[2]))

class Label(Enum):
    HUMAN_SPEECH = 1
    URBAN_NOISE = 2

def cross_correlated_average(dataset_ffts):
    signal = dataset_ffts[0]

    for i in  range(1, len(dataset_ffts)):
        correlation = np.correlate(signal, dataset_ffts[i] * i, mode='full')
        max_correlation = np.argmax(correlation)
        dataset_ffts[i] = np.roll(dataset_ffts[i], max_correlation)
        signal += dataset_ffts[i]

    signal /= len(dataset_ffts)
    return signal


def speech_stats():
    window_size = 8000 #1s
    datasets = [
        ('training_dataset/humans_speaking01', Label.HUMAN_SPEECH),
        ('training_dataset/humans_speaking02', Label.HUMAN_SPEECH),
        ('training_dataset/humans_speaking_female01', Label.HUMAN_SPEECH),
        ('training_dataset/humans_speaking_mix', Label.HUMAN_SPEECH),
        ('training_dataset/random_urban_noises', Label.URBAN_NOISE),
        ('training_dataset/random_noises_mix', Label.URBAN_NOISE),
    ]

    human_ffts = []
    non_human_ffts = []

    for (dataset_name, label) in datasets:
        print(f"Processing dataset {dataset_name}, with label {label}")

        dataset_ffts = handle_audio_file_dataset(get_audio_file_paths_in_dir(dataset_name), window_size)

        flatten_ffts_windows = arr2d_to_arr1d(dataset_ffts);

        dataset_mean_vector, _ = compute_mean_and_std(flatten_ffts_windows)

        normalized_energy_density = compute_spectral_density_normalized(dataset_mean_vector)
        if label == Label.HUMAN_SPEECH:
            human_ffts.append(normalized_energy_density)
        else:
            non_human_ffts.append(normalized_energy_density)

    average_signal = cross_correlated_average(human_ffts)

    to_visualize = np.array([average_signal, *non_human_ffts])

    visualize_plots(to_visualize, "average human vs average non-human sound")

    plt.pause(1000)

if __name__ == "__main__":
    speech_stats()
