from enum import Enum
from pydub import AudioSegment
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import os

from utils import supported_formats, get_audio_file_paths_in_dir, resample_audio, read_audio_file, compute_fft, visualize_plots, visualize_windows, compute_mean_and_std

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

def test():
    window_size = 8000 #1s
    step_size = 8000

    datasets = [
        ('test_samples/humans_speaking01', Label.HUMAN_SPEECH),
        ('test_samples/humans_speaking02', Label.HUMAN_SPEECH),
        ('test_samples/humans_speaking_mix', Label.HUMAN_SPEECH),
        ('test_samples/random_urban_noises', Label.URBAN_NOISE),
        ('test_samples/random_noises_mix', Label.URBAN_NOISE),
    ]

    plots = []

    for (dataset_name, label) in datasets:
        print(f"Processing dataset {dataset_name}, with label {label}")

        dataset_ffts = handle_audio_file_dataset(get_audio_file_paths_in_dir(dataset_name), window_size)

        flatten_dataset_windows = arr2d_to_arr1d(dataset_ffts);
#        visualize_windows(flatten_dataset_windows, window_size, "visualize dataset: " + dataset_name)
        dataset_mean_vector, dataset_std_dev_norm = compute_mean_and_std(flatten_dataset_windows)
        plots.append((dataset_mean_vector, dataset_std_dev_norm, label))

    visualize_plots(plots, "Mean vectors and standard deviation")

#    another_dataset_human_audio = 'test_samples/female_speaker_en.mp3';
#
#    another_dataset_human_ffts = handle_audio_file(another_dataset_human_audio, window_size)
#    print(another_dataset_human_ffts.shape)
#
#    visualize_windows(another_dataset_human_ffts, window_size, "Random windows from another dataset")



#    another_dataset_human_spectrum = split_audio_to_windows(another_dataset_human_audio, window_size, step_size)

#    print(len(humans_speech_mean_vector))
#    print(len(urban_noises_mean_vector))

#    visualize_windows(np.vstack((humans_speech_mean_vector, urban_noises_mean_vector)), window_size, "Humans speech mean vector")
#    visualize_windows(np.vstack((urban_noises_mean_vector, another_dataset_human_spectrum)), window_size, "Urban noises mean vector")
    plt.pause(1000)

if __name__ == "__main__":
    test()
