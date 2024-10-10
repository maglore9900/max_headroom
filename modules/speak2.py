
import noisereduce as nr
import numpy as np
import pyaudio
import speech_recognition as sr
import pyttsx3
import os
import random
import urllib.parse
import requests
from pydub import AudioSegment
import io
import wave
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Speak:
    def __init__(self, env):
        self.url = env("STREAM_SPEAK_URL")
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.model_name = env("LISTEN_MODEL".lower(), default="whisper")
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.noise_threshold = 500  # Initial placeholder for noise threshold
        self.recent_noise_levels = deque(maxlen=30)  # Track recent noise levels for dynamic adjustment
        self.voice = env("ALL_TALK_VOICE")
        self.silence = int(env("TIME_SILENCE"))

        # Initialize transcription models
        if self.model_name == "whisper":
            from faster_whisper import WhisperModel
            self.whisper_model_path = "large-v2"
            self.whisper_model = WhisperModel(self.whisper_model_path, device="cuda")  # Nvidia GPU mode
        else:
            self.recognizer = sr.Recognizer()

    def adjust_noise_threshold(self, audio_chunk):
        """Dynamically adjust the noise threshold based on the ambient noise levels of the current chunk."""
        noise_level = np.abs(audio_chunk).mean()
        self.recent_noise_levels.append(noise_level)
        
        # Calculate a new threshold based on recent noise levels (running average)
        self.noise_threshold = np.mean(self.recent_noise_levels)

    def listen_to_microphone(self):
        """Function to listen to the microphone input and return raw audio data after applying dynamic noise reduction."""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        stream.start_stream()
        print("Listening...")

        audio_data = b""
        silence_duration = self.silence  # Time of silence in seconds before stopping
        silence_counter = 0
        detected_speech = False
        
        while True:
            data = stream.read(self.chunk_size)
            audio_data += data

            # Convert to numpy array for noise reduction and dynamic adjustment
            np_data = np.frombuffer(data, dtype=np.int16)
            
            # Adjust noise threshold dynamically using the current chunk
            self.adjust_noise_threshold(np_data)
            
            # Reduce noise in the current chunk
            reduced_noise_data = nr.reduce_noise(y=np_data, sr=self.sample_rate)
            
            # Check if speech is detected based on the dynamically adjusted noise threshold
            if np.abs(reduced_noise_data).mean() > self.noise_threshold:
                detected_speech = True
                silence_counter = 0  # Reset silence counter when speech is detected
            elif detected_speech:  # If we already detected speech and now there is silence
                silence_counter += self.chunk_size / self.sample_rate
                if silence_counter >= silence_duration:
                    print("Silence detected. Stopping.")
                    break
        
        stream.stop_stream()
        stream.close()
        p.terminate()

        return audio_data

    def transcribe(self):
        """
        Function to transcribe audio from the microphone. Stops when no speech is detected.
        """
        print("Listening until silence is detected.")

        audio_data = self.listen_to_microphone()

        # Transcription logic here
        if self.model_name == "whisper":
            energy_threshold = 0.0001
            audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
            energy = np.mean(np.abs(audio_np))
            if energy > energy_threshold:
                segments, _ = self.whisper_model.transcribe(audio_np, beam_size=5)
                transcription = " ".join([segment.text for segment in segments])
                print(f"Whisper Transcription: {transcription}")
                return transcription
        else:
            with self.microphone as source:
                try:
                    audio = sr.AudioData(audio_data, self.sample_rate, 2)
                    transcription = self.recognizer.recognize_google(audio)
                    print(f"Google Transcription: {transcription}")
                    return transcription
                except:
                    pass
                
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
                      
    def stream(self, text):
        # Example parameters
        voice = self.voice
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

                if stream is None:
                    # Define stream parameters without any modifications
                    stream = p.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=audio_segment.frame_rate,
                                    output=True)

                # Play the original chunk (without any modification)
                stream.write(audio_segment.raw_data)

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
        

        
