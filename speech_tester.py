import numpy as np
from spectrogram_extractor import arr2d_to_arr1d, cross_correlate_shifts, handle_audio_file_dataset

from utils import compute_spectral_density_normalized, get_audio_file_paths_in_dir, read_audio_file, compute_fft, visualize_plots, compute_mean_and_std

from spectrogram_extractor import split_audio_to_windows

def cos_similarity(lhs, rhs):
    cos_similarity = np.dot(rhs, lhs) / (np.linalg.norm(rhs) * np.linalg.norm(lhs))

    return cos_similarity
    
def classifier(fft_sample, reference_fft, threshold):
    normalized_energy_density = compute_spectral_density_normalized(fft_sample)
    normalized_reference_energy_density = compute_spectral_density_normalized(reference_fft)

    cos_sim = cos_similarity(normalized_energy_density, normalized_reference_energy_density)

    print(cos_sim)
    return cos_sim > threshold

def classify_fft_window_matrix(fft_window_matrix, reference_fft, threshold=0.7):
    classification_matrix = np.array([[classifier(window, reference_fft, threshold) for window in audio_window_array] for audio_window_array in fft_window_matrix])

    return classification_matrix


if __name__ == "__main__":
    window_size = 8000

    reference_fft_file = "reference_fft.npy"

    human_test_dataset = "testing_dataset/human_speech"
    non_human_test_dataset = "testing_dataset/non_human_speech"

    reference_fft = np.load(reference_fft_file)

    human_dataset_ffts = handle_audio_file_dataset(get_audio_file_paths_in_dir(human_test_dataset), window_size)
#    non_human_dataset_ffts = handle_audio_file_dataset(get_audio_file_paths_in_dir(non_human_test_dataset), window_size)

    flattened_human_dataset_ffts = arr2d_to_arr1d(human_dataset_ffts)
#    flattened_non_human_dataset_ffts = arr2d_to_arr1d(non_human_dataset_ffts)

    print("Human dataset shape: ", np.sum(flattened_human_dataset_ffts))

    human_dataset_classification = classify_fft_window_matrix(flattened_human_dataset_ffts, reference_fft, 0.7)
#    non_human_dataset_classification = classify_fft_window_matrix(flattened_non_human_dataset_ffts, reference_fft, 0.7)

    print(human_dataset_classification)

    human_avg = np.average(human_dataset_classification)
#    non_human_avg = np.average(non_human_dataset_classification)

    print("Human avg: ", human_avg)
#    print("Non human avg: ", non_human_avg)
