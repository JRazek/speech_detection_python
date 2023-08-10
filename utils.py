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

def compute_fft(samples):
    spectrum = np.abs(np.fft.fft(samples))
    positive_frequencies = spectrum[:len(samples) // 2]  # Only keep the positive frequencies
    return positive_frequencies

def compute_spectral_density_normalized(ffts):
    f_pds = np.abs(ffts / len(ffts))**2 / len(ffts) / 8000
    f_rms = np.sqrt(np.sum(f_pds**2) / len(ffts) * 8000)
    return f_pds

def compute_mean_and_std(matrix):
    matrix = np.vstack(matrix)

    # Calculate mean vector
    mean_vector = np.mean(matrix, 0)

    # Calculate standard deviation vector
    std_dev = np.std(matrix, 0)

    std_dev_vector_norm = np.linalg.norm(std_dev)

    return mean_vector, std_dev_vector_norm

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
    plt.tight_layout()

def visualize_plots(human_spectrums, human_correls, non_human_spectrums, non_human_correls, human_reference_spectrum):
    # Determine the number of rows we will have
    num_rows = max(len(human_spectrums), len(non_human_spectrums))

    plt.figure(figsize=(20, 4 * num_rows))  # Adjusting figure size based on number of plots

    # Plotting human spectrums
    for idx, (plot, correl) in enumerate(zip(human_spectrums, human_correls)):
        ax = plt.subplot(num_rows, 2, 2*idx + 1)
        ax.text(0.05, 0.85, "Human spectrum " + str(idx + 1), transform=ax.transAxes, color='blue', backgroundcolor='white')
        ax.text(0.05, 0.75, "Correlation: " + str(correl), transform=ax.transAxes, color='blue', backgroundcolor='white')
        plt.plot(plot)
        plt.plot(human_reference_spectrum, color='red', alpha=0.5)  # Overlaying with human_extracted_spectrum

    # Plotting non-human spectrums
    for idx, (plot, correl) in enumerate(zip(non_human_spectrums, non_human_correls)):
        ax = plt.subplot(num_rows, 2, 2*idx + 2)
        ax.text(0.05, 0.85, "Non-human spectrum " + str(idx + 1), transform=ax.transAxes, color='blue', backgroundcolor='white')
        ax.text(0.05, 0.75, "Correlation: " + str(correl), transform=ax.transAxes, color='blue', backgroundcolor='white')
        plt.plot(plot)
        plt.plot(human_reference_spectrum, color='red', alpha=0.5)  # Overlaying with human_extracted_spectrum

    plt.tight_layout()  # Adjust spacing between subplots for clarity
