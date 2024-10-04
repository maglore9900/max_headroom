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
        self.noise_threshold = 500
        
        
        # Initialize transcription models
        if self.model_name == "whisper":
            from faster_whisper import WhisperModel
            self.whisper_model_path = "large-v2"
            self.whisper_model = WhisperModel(self.whisper_model_path, device="cuda")  # Mvidia GPU mode
            # self.whisper_model = WhisperModel(self.whisper_model_path, device="cpu")  # CPU mode
        else:
            self.recognizer = sr.Recognizer()

    def listen_to_microphone(self, time_listen=10):
        """Function to listen to the microphone input and return raw audio data."""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        stream.start_stream()
        print("Listening...")

        audio_data = b""
        ambient_noise_data = b""
        
        try:
            for i in range(int(self.sample_rate / self.chunk_size * time_listen)):
                audio_chunk = stream.read(self.chunk_size)
                audio_data += audio_chunk

                # Capture ambient noise in the first 2 seconds
                if i < int(self.sample_rate / self.chunk_size * 1):  # First 1 seconds
                    ambient_noise_data += audio_chunk

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        return audio_data, ambient_noise_data
    
    def apply_noise_cancellation(self, audio_data, ambient_noise):
        """Apply noise cancellation to the given audio data, using ambient noise from the first 2 seconds."""
        # Convert to NumPy array (normalize to [-1, 1])
        audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
        ambient_noise_np = np.frombuffer(ambient_noise, np.int16).astype(np.float32) / 32768.0

        # Use ambient noise as noise profile
        reduced_noise = nr.reduce_noise(y=audio_np, sr=self.sample_rate, y_noise=ambient_noise_np)

        # Convert back to int16 after noise reduction for compatibility with Whisper
        reduced_noise_int16 = (reduced_noise * 32768).astype(np.int16)

        return reduced_noise_int16.tobytes()  # Return as bytes

    def transcribe(self, audio_data):
        """Transcribe the audio data using the selected model."""
        if self.model_name == "whisper":
            # Whisper expects float32 audio data
            energy_threshold = 0.0001
            # Convert int16 PCM audio data to float32
            audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0

            # Calculate energy of the audio to determine if it should be transcribed
            energy = np.mean(np.abs(audio_np))

            # Only transcribe if energy exceeds the threshold
            if energy > energy_threshold:
                # Transcribe using Whisper model (assumed to be already loaded in self.whisper_model)
                segments, _ = self.whisper_model.transcribe(audio_np, beam_size=5)
                transcription = " ".join([segment.text for segment in segments])
                print(f"Whisper Transcription: {transcription}")
                return transcription
            else:
                print("Audio energy below threshold; no transcription performed.")
                return ""
        else:
            # Google SpeechRecognition code (no changes here)
            recognizer = sr.Recognizer()
            audio_buffer = io.BytesIO()

            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Assuming mono audio
                wav_file.setsampwidth(2)  # Assuming 16-bit audio
                wav_file.setframerate(16000)  # Assuming 16kHz sample rate
                wav_file.writeframes(audio_data)  # Write raw PCM data

            # Reset the buffer's position to the start
            audio_buffer.seek(0)

            # Use SpeechRecognition's AudioFile to handle the in-memory WAV file
            with sr.AudioFile(audio_buffer) as source:
                audio = recognizer.record(source)
                try:
                    transcription = recognizer.recognize_google(audio)
                    print(f"Google Transcription: {transcription}")
                    return transcription
                except sr.UnknownValueError:
                    print("Google could not understand audio")
                    return ""
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    return ""


    def listen(self, time_listen=8):
        """Main transcoder function that handles listening, noise cancellation, and transcription."""
        # Listen to the microphone and get both raw audio and ambient noise
        raw_audio, ambient_noise = self.listen_to_microphone(time_listen)
        
        # Apply noise cancellation using the ambient noise from the first 2 seconds
        clean_audio = self.apply_noise_cancellation(raw_audio, ambient_noise=ambient_noise)
        
        # Transcribe the clean audio
        transcription = self.transcribe(clean_audio)
        
        return transcription

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
        voice = ""
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

            
# Example usage:
# sp = Speak(model="whisper")  # or "whisper" or "google"
# transcription = sp.transcoder(time_listen=10)
# print("Final Transcription:", transcription)
