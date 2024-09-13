import os
import pyaudio
import numpy as np
import noisereduce as nr
from faster_whisper import WhisperModel
from numpy import frombuffer, int16
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Speak:
    def __init__(self):
        self.model_path = "large-v2"  # Use the appropriate faster-whisper model path
        self.model = WhisperModel(self.model_path, device="cuda")
        self.sample_rate = 16000
        self.channels = 1
        self.chunk = 1024  # Number of frames per buffer
        self.noise_threshold = 500  # Threshold to detect ambient noise

    def listen3(self, duration=5):
        """ Listens to the microphone for a specific duration and transcribes the audio using faster-whisper, with noise suppression """
        p = pyaudio.PyAudio()

        # print(f"Listening for {duration} seconds...")

        # Open a stream to capture audio input from the microphone
        stream = p.open(format=pyaudio.paInt16, 
                        channels=self.channels, 
                        rate=self.sample_rate, 
                        input=True, 
                        frames_per_buffer=self.chunk)

        frames = []

        for _ in range(0, int(self.sample_rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            audio_data = frombuffer(data, dtype=int16)

            # Apply noise reduction only if there's valid audio data
            if np.any(audio_data):  # Check if audio data contains non-zero values
                reduced_noise_data = nr.reduce_noise(y=audio_data, sr=self.sample_rate)

                # Calculate RMS value, ensuring no invalid data (NaN) is used
                rms_value = np.sqrt(np.mean(np.square(reduced_noise_data)))

                # Only add frames that are below the noise threshold (i.e., filter out ambient noise)
                if not np.isnan(rms_value) and rms_value < self.noise_threshold:
                    frames.append(reduced_noise_data.astype(int16).tobytes())
            else:
                print("Invalid or zero audio data encountered.")

        # Stop and close the audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Combine the audio frames into a single array for transcription
        if frames:
            audio_data = np.frombuffer(b"".join(frames), dtype=int16)

            # Transcribe the audio using faster-whisper
            segments, info = self.model.transcribe(audio_data)

            # Output the transcription
            for segment in segments:
                print(f"Transcription: {segment.text}")
        else:
            print("No valid audio data for transcription due to ambient noise.")

if __name__ == "__main__":
    sp = Speak()
    sp.listen3(duration=5)  # Listen for 5 seconds