from pydub import AudioSegment
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import os

from utils import supported_formats, get_audio_file_paths_in_dir, resample_audio, read_audio_file, compute_spectrogram, visualize_windows

base_frequency = 8000

def read_spectrograms(file_path, window_size, step_size):
    resampled_audio = read_audio_file(file_path, base_frequency)
    resampled_audio_raw = np.array(resampled_audio.get_array_of_samples())
    spectrogram = compute_spectrogram(resampled_audio_raw, window_size, step_size)
    return spectrogram

def get_random_windows_from_spectrums(spectrums):
    random_windows = []
    for spectrum in spectrums:
        random_window = np.random.randint(0, spectrum.shape[1])
        random_windows.append(spectrum[:, random_window])
    return random_windows

def read_spectrums(files, window_size, step_size):
    spectrums = []
    for file in files:
        spectrums.append(read_spectrograms(file, window_size, step_size))
    return spectrums

if __name__ == "__main__":
    window_size = 8000 #1s
    step_size = 8000

    files = get_audio_file_paths_in_dir('test_samples/humans_speaking')
    spectrums = read_spectrums(files, window_size, step_size)

    visualize_windows(get_random_windows_from_spectrums(spectrums), window_size, "Random windows from spectrums")

    plt.pause(10000)
