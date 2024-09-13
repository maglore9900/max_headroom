import noisereduce as nr
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
from faster_whisper import WhisperModel
import speech_recognition as sr
import pyttsx3
import os
import random
from pydub import AudioSegment
import urllib.parse
import requests
import json
# from numpy import frombuffer, int16

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Speak:
    def __init__(self, model="whisper"):
        self.url = "http://127.0.0.1:7851/api/tts-generate"
        
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.model_name = model
        self.sample_rate = 16000
        self.chunk_size = 1024
        
        self.noise_threshold = 500  # Threshold to detect ambient noise
        
        # Initialize Vosk and Whisper models
        if self.model_name == "vosk":
            self.model_path = os.path.join(os.path.dirname(__file__), "../models/vosk-model-en-us-0.42-gigaspeech")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, 16000)
        elif self.model_name == "whisper":
            self.whisper_model_path = "large-v2"
            self.recognizer = WhisperModel(self.whisper_model_path, device="cuda")  # Adjust if you don't have a CUDA-compatible GPU
            # self.recognizer = None
        else:
            self.recognizer = sr.Recognizer()

    def listen3(self, time_listen=10):
        """
        Streams audio from the microphone and applies noise cancellation.
        """
        counter = 0
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        stream.start_stream()
        print("Listening...")
        
        try:
            while counter < time_listen:
                # Read audio data from the stream
                audio_data = stream.read(8000, exception_on_overflow=False)
                # Convert the audio data to a numpy array of int16
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                # Apply noise reduction
                reduced_noise = nr.reduce_noise(y=audio_np, sr=self.sample_rate)
                # Calculate RMS to detect ambient noise levels
                rms_value = np.sqrt(np.mean(np.square(reduced_noise)))
                if rms_value < self.noise_threshold:
                    # Pass the reduced noise (still in numpy format) to the transcoder
                    self.transcoder(reduced_noise.tobytes())
                else:
                    print(f"Ambient noise detected: RMS {rms_value} exceeds threshold {self.noise_threshold}")
                counter += 1
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Clean up the stream resources
            stream.stop_stream()
            stream.close()
            p.terminate()

    def transcoder(self, audio_data):
        """
        Transcodes audio data to text using the specified model.
        """
        if self.model_name == "vosk":
            if self.recognizer.AcceptWaveform(audio_data):
                    result = json.loads(self.recognizer.Result())
                    if result["text"]:
                        print(f"Recognized: {result['text']}")
                        return result['text']
                    return result
        elif self.model_name == "whisper":

            result, _ = self.recognizer.transcribe(audio_data, beam_size=5)
            return result['text']
        else:
            result = self.recognizer.recognize_google(audio_data)
            return result

        
    # def vosk_transcription(self):
    #     """
    #     Handles Vosk-based transcription of streamed audio with noise cancellation.
    #     """
    #     recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
    #     stream = self.stream_with_noise_cancellation()
        
    #     for audio_chunk in stream:
    #         if recognizer.AcceptWaveform(audio_chunk):
    #             result = recognizer.Result()
    #             print(result)  # Handle or process the transcription result

    # def whisper_transcription(self):
    #     """
    #     Handles Faster-Whisper-based transcription of streamed audio with noise cancellation.
    #     """
    #     stream = self.stream_with_noise_cancellation()
        
    #     for audio_chunk in stream:
    #         # Transcribe the cleaned audio using faster-whisper
    #         result, _ = self.whisper_model.transcribe(audio_chunk, beam_size=5)
    #         print(result['text'])  # Handle or process the transcription result
            
    # def listen(self):
    #     if self.model == "vosk":
    #         self.vosk_transcription()
    #     elif self.model == "whisper":
    #         self.whisper_transcription()
    #     else:
    #         raise ValueError("Invalid model specified. Please specify either 'vosk' or 'whisper'.")

    def glitch_stream_output(self, text):
        def change_pitch(sound, octaves):
            val = random.randint(0, 10)
            if val == 1:
                new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
                return sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)
            else:
                return sound

        def convert_audio_format(sound, target_sample_rate=16000):
            # Ensure the audio is in PCM16 format
            sound = sound.set_sample_width(2)  # PCM16 = 2 bytes per sample
            # Resample the audio to the target sample rate
            sound = sound.set_frame_rate(target_sample_rate)
            return sound

        # Example parameters
        voice = "maxheadroom_00000045.wav"
        language = "en"
        output_file = "stream_output.wav"
        
        # Encode the text for URL
        encoded_text = urllib.parse.quote(text)
        
        # Create the streaming URL
        streaming_url = f"http://localhost:7851/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        try:
            # Stream the audio data
            response = requests.get(streaming_url, stream=True)
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            stream = None
            
            # Process the audio stream in chunks
            chunk_size = 1024 * 6  # Adjust chunk size if needed
            audio_buffer = b''

            for chunk in response.iter_content(chunk_size=chunk_size):
                audio_buffer += chunk

                if len(audio_buffer) < chunk_size:
                    continue
                
                audio_segment = AudioSegment(
                    data=audio_buffer,
                    sample_width=2,  # 2 bytes for 16-bit audio
                    frame_rate=24000,  # Assumed frame rate, adjust as necessary
                    channels=1  # Assuming mono audio
                )

                # Randomly adjust pitch
                octaves = random.uniform(-0.1, 1.5)
                modified_chunk = change_pitch(audio_segment, octaves)

                if random.random() < 0.001:  # 1% chance to trigger stutter
                    repeat_times = random.randint(2, 5)  # Repeat 2 to 5 times
                    for _ in range(repeat_times):
                        stream.write(modified_chunk.raw_data)

                # Convert to PCM16 and 16kHz sample rate after the stutter effect
                modified_chunk = convert_audio_format(modified_chunk, target_sample_rate=16000)

                if stream is None:
                    # Define stream parameters
                    stream = p.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=modified_chunk.frame_rate,
                                    output=True)

                # Play the modified chunk
                stream.write(modified_chunk.raw_data)

                # Reset buffer
                audio_buffer = b''

            # Final cleanup
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
        except:
            self.engine.say(text)
            self.engine.runAndWait()
# Example usage:
# sp = Speak(vosk_model_path="path_to_vosk_model", whisper_model_path="large-v2")
# sp.vosk_transcription()  # To start Vosk transcription
# sp.whisper_transcription()  # To start Faster-Whisper transcription
sp = Speak()
# sp.glitch_stream_output("Hello, world!")
sp.listen3()