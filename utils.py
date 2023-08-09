from pydub import AudioSegment
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
import os

supported_formats = ['wav', 'mp3', 'flac', 'ogg']

def get_audio_file_paths_in_dir(directory, dst_freq=-1):
    all_files = os.listdir(directory)

    # Filter out the audio files based on their extensions
    audio_files = [os.path.join(directory, file) for file in all_files if file.split('.')[-1] in supported_formats]

    return audio_files

def resample_audio(audio, dst_freq):
    resampled_audio = audio.set_frame_rate(dst_freq).set_channels(1)
    return resampled_audio

def read_audio_file(file_path, dst_freq=-1):
    format = file_path.split('.')[-1]
    if format not in supported_formats:
        raise ValueError(f"Unsupported audio format: {format}")

    audio = AudioSegment.from_file(file_path, format=format)

    if dst_freq == -1:
        return audio

    audio = resample_audio(audio, dst_freq)
    return audio

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
