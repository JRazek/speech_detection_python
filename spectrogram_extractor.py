from pydub import AudioSegment
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import os

supported_formats = ['wav', 'mp3', 'flac', 'ogg']
base_frequency = 8000

def get_audio_files_in_dir(directory):
    all_files = os.listdir(directory)

    # Filter out the audio files based on their extensions
    audio_files = [os.path.join(directory, file) for file in all_files if file.split('.')[-1] in supported_formats]

    return audio_files

# Step 1: Resample the audio
def resample_audio(file_path):
    format = file_path.split('.')[-1]
    if format not in supported_formats:
        raise ValueError(f"Unsupported audio format: {format}")

    audio = AudioSegment.from_file(file_path, format=format)
    audio = audio.set_frame_rate(base_frequency).set_channels(1)
    samples = np.array(audio.get_array_of_samples())
    return samples

# Step 2: Compute the FFT spectrogram
def compute_spectrogram(samples, window_size, step_size):
    window = np.hanning(window_size)
    start = 0
    spectrogram = []

    while start + window_size <= len(samples):
        segment = samples[start:start+window_size]
        windowed_segment = segment * window
        spectrum = np.abs(fft(windowed_segment))
        spectrogram.append(spectrum[:window_size // 2])  # we only keep the positive frequencies
        start += step_size

    return np.array(spectrogram).T  # Transpose to have time on the x-axis

def read_spectrogram(file_path, window_size, step_size):
    resampled_samples = resample_audio(file_path)
    spectrogram = compute_spectrogram(resampled_samples, window_size, step_size)
    return spectrogram

def visualize_windows(spectra, window_size, figure_name="FFT Spectrum"):
    # Generate frequency axis
    freqs = np.linspace(0, 8000 / 2, window_size // 2)

    plt.figure(figsize=(10, 4))

    # Get a set of colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(spectra)))

    for idx, spectrum in enumerate(spectra):
        plt.plot(freqs, spectrum, color=colors[idx], label=f"Spectrum {idx + 1}")

    plt.title(figure_name)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()  # To show the label for each spectrum
    plt.tight_layout()
    plt.show(block=False)

def get_random_windows_from_spectrums(spectrums):
    random_windows = []
    for spectrum in spectrums:
        random_window = np.random.randint(0, spectrum.shape[1])
        random_windows.append(spectrum[:, random_window])
    return random_windows

def read_spectrums(files, window_size, step_size):
    spectrums = []
    for file in files:
        spectrums.append(read_spectrogram(file, window_size, step_size))
    return spectrums

if __name__ == "__main__":
    window_size = 8000 #1s
    step_size = 8000

    files = get_audio_files_in_dir('test_samples/humans_speaking')
    spectrums = read_spectrums(files, window_size, step_size)

    visualize_windows(get_random_windows_from_spectrums(spectrums), window_size, "Random windows from spectrums")

    plt.pause(10000)
