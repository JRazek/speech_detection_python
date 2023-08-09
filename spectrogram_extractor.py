from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from utils import get_audio_file_paths_in_dir, read_audio_file, compute_fft, visualize_plots, compute_mean_and_std

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

def extract_dataset_window_matrix(dataset_path, window_size):
    audio_files = np.asarray(dataset_path)

    audio_matrix = read_audio_files(audio_files)

    audio_windows_matrix = np.array([split_audio_to_windows(seq, window_size) for seq in audio_matrix])

    return audio_windows_matrix

def handle_audio_file_dataset(audio_files, window_size):
    audio_windows_matrix = extract_dataset_window_matrix(audio_files, window_size)

    ffts = np.array([[compute_fft(window) for window in audio_window_array] for audio_window_array in audio_windows_matrix])

    return ffts

def handle_audio_file(audio_file, window_size):
    return arr2d_to_arr1d(handle_audio_file_dataset([audio_file], window_size))


def arr2d_to_arr1d(arr2d):
    return np.reshape(arr2d, (arr2d.shape[0] * arr2d.shape[1], arr2d.shape[2]))

class Label(Enum):
    HUMAN_SPEECH = 1
    URBAN_NOISE = 2

def cross_correlate_shifts(dataset_ffts, reference_spectrum):
    aligned_ffts = []  # 

    for i in range(0, len(dataset_ffts)):
        print("shape of dataset_ffts[i]: ", dataset_ffts[i].shape)
        print("shape of reference_spectrum: ", reference_spectrum.shape)

        correlation = np.correlate(reference_spectrum, dataset_ffts[i], mode='full')
        print(f"Correlation for fft_{i}: {correlation}")

        argmax = np.argmax(correlation)
        argmin = np.argmin(correlation)

        print(f"fft_{i}, argmax: {argmax}, max: {correlation[argmax]}")
        print(f"fft_{i}, argmin: {argmin}, min: {correlation[argmin]}")

        shift = len(reference_spectrum) - argmax - 1

        shifted_fft = np.roll(dataset_ffts[i], -shift)

        aligned_ffts.append(shifted_fft)

    return np.array(aligned_ffts)

def cross_correlated_average(dataset_ffts):
    if len(dataset_ffts) == 1:
        return dataset_ffts[0]
    aligned_ffts = cross_correlate_shifts(dataset_ffts[1:], dataset_ffts[0])
    accumulative_signal = aligned_ffts.sum(axis=0)
    accumulative_signal /= len(aligned_ffts)
    return accumulative_signal

def speech_stats():
    window_size = 8000 #1s
    datasets = [
        ('training_dataset/humans_speaking02', Label.HUMAN_SPEECH),
        ('training_dataset/humans_speaking01', Label.HUMAN_SPEECH),
        ('training_dataset/humans_speaking_female01', Label.HUMAN_SPEECH),
        ('training_dataset/humans_speaking_mix', Label.HUMAN_SPEECH),
        ('testing_dataset/human_speech', Label.HUMAN_SPEECH),
        ('training_dataset/random_urban_noises', Label.URBAN_NOISE),
        ('training_dataset/random_noises_mix', Label.URBAN_NOISE),
#        ('training_dataset/sine440_1', Label.HUMAN_SPEECH),
#        ('training_dataset/sine440_2', Label.HUMAN_SPEECH),
#        ('training_dataset/sine440_3', Label.URBAN_NOISE),
    ]

    human_ffts = []
    non_human_ffts = []

    for (dataset_name, label) in datasets:
        print(f"Processing dataset {dataset_name}, with label {label}")

        dataset_ffts = handle_audio_file_dataset(get_audio_file_paths_in_dir(dataset_name), window_size)

        flatten_ffts_windows = arr2d_to_arr1d(dataset_ffts);

        dataset_mean_vector, _ = compute_mean_and_std(flatten_ffts_windows)

        dataset_mean_vector = dataset_mean_vector / np.std(dataset_mean_vector)

        if label == Label.HUMAN_SPEECH:
            human_ffts.append(dataset_mean_vector)
        else:
            non_human_ffts.append(dataset_mean_vector)

    reference_fft = cross_correlated_average(human_ffts) #note that this is cross corr average of ALREADE SPEC DENSITY NORMALIZED

    human_ffts_shifted = cross_correlate_shifts(human_ffts, reference_fft)
    non_human_ffts_shifted = cross_correlate_shifts(non_human_ffts, reference_fft)

    visualize_plots(human_ffts, reference_fft, non_human_ffts)
    visualize_plots(human_ffts_shifted, reference_fft, non_human_ffts_shifted)

    np.save("reference_fft.npy", reference_fft)

    plt.pause(10000)
    plt.savefig("speech_stats.svg")

if __name__ == "__main__":
    speech_stats()
