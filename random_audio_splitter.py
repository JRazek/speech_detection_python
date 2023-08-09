from pydub import AudioSegment
import random
import os

base_freq = 8000

from utils import read_audio_file

def split_audio_randomly(audio, window_length, num_windows):
    # Calculate maximum start point
    max_start = len(audio) - window_length

    segments = []

    for i in range(num_windows):
        # Randomly choose a start point
        start_point = random.randint(0, max_start)
        end_point = start_point + window_length

        # Extract the window from the audio
        segment = audio[start_point:end_point]

        segments.append(segment)

    return segments

if __name__ == "__main__":
    window_size = 8000 #1s

    input_file = "test_samples/urban_new_york_noise.mp3"
    output_files_dir = "test_samples/random_urban_noises/"

    src_audio_resampled = read_audio_file(input_file, base_freq)

    segments = split_audio_randomly(src_audio_resampled, window_size, 10)

    for idx, segment in enumerate(segments):
        output_file = os.path.join(output_files_dir, f"window_{idx+1}.mp3")
        segment.export(output_file, format="mp3")
