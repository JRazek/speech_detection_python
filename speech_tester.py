import sounddevice as sd
import numpy as np
from io import BytesIO
from pydub import AudioSegment

def classifier(duration, samplerate=8000):
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()  # Wait for the recording to finish

    # Convert the numpy array to bytes
    audio_bytes = audio_data.tobytes()

    # Convert bytes to an in-memory binary stream
    byte_stream = BytesIO(audio_bytes)

    # Create an AudioSegment from the byte stream
    audio_segment = AudioSegment.from_raw(byte_stream, sample_width=2, frame_rate=samplerate, channels=2)

    return audio_segment
