import requests
import winsound
import speech_recognition as sr
import pyttsx3 
import os
import vlc
import time
import pyaudio
from pydub import AudioSegment
import random
import urllib.parse

import os
import json
import pyaudio
# from vosk import Model, KaldiRecognizer 
import noisereduce as nr
from numpy import frombuffer, int16
import numpy as np

from faster_whisper import WhisperModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Speak:
    def __init__(self):
        self.url = "http://127.0.0.1:7851/api/tts-generate"
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # self.model_path = os.path.join(os.path.dirname(__file__), "../models/vosk-model-en-us-0.42-gigaspeech")
        # self.model = Model(self.model_path)
        # self.recognizer = KaldiRecognizer(self.model, 16000)
        
        self.model_path = "large-v2"  # Use the appropriate faster-whisper model path
        self.model = WhisperModel(self.model_path, device="cuda")
        self.sample_rate = 16000
        self.channels = 1
        self.chunk = 1024  # Number of frames per buffer
        self.noise_threshold = 500  # Threshold to detect ambient noise
        
      
    #! listen with google  
    def listen(self):
        with self.microphone as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            try:
                # Listen with a 5-second timeout
                audio = self.recognizer.listen(source, timeout=10)
                try:
                    text = self.recognizer.recognize_google(audio)
                    print("You said: ", text)
                    return text
                except sr.UnknownValueError:
                    print("Sorry, I didn't get that.")
                    return None
                except sr.RequestError as e:
                    print("Sorry, I couldn't request results; {0}".format(e))
                    return None
            except sr.WaitTimeoutError:
                print("Timeout. No speech detected.")
                return None  

    # #! listen with vosk
    # def listen2(self, time_listen=15):
    #     noise_threshold=500
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    #     stream.start_stream()
    #     print("Listening...")
    #     count = 0
    #     try:
    #         while count < time_listen:
    #             data = stream.read(8000, exception_on_overflow=False)
    #             filtered_data = nr.reduce_noise(y=frombuffer(data, dtype=int16), sr=16000).astype(int16).tobytes()

    #             # Calculate RMS to detect ambient noise levels
    #             rms_value = np.sqrt(np.mean(np.square(np.frombuffer(filtered_data, dtype=int16))))

    #             if rms_value < noise_threshold:
    #                 if self.recognizer.AcceptWaveform(filtered_data):
    #                     result = json.loads(self.recognizer.Result())
    #                     if result["text"]:
    #                         print(f"Recognized: {result['text']}")
    #                         return result['text']
    #             else:
    #                 print(f"Ambient noise detected: RMS {rms_value} exceeds threshold {noise_threshold}")
    #             count += 1
    #     except KeyboardInterrupt:
    #         print("Stopping...")
    #     finally:
    #         stream.stop_stream()
    #         stream.close()
    #         p.terminate()
            
    #! Listen with Faster Whisper
    def listen3(self, duration=10):
        """ Listens to the microphone for a specific duration and transcribes the audio using faster-whisper, with noise suppression """
        p = pyaudio.PyAudio()

        print("Listening...")

        # Open a stream to capture audio input from the microphone
        stream = p.open(format=pyaudio.paInt16, 
                        channels=self.channels, 
                        rate=self.sample_rate, 
                        input=True, 
                        frames_per_buffer=self.chunk)

        frames = []
        transcribed_text = []

        for _ in range(0, int(self.sample_rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            audio_data = frombuffer(data, dtype=int16)

            # Apply noise reduction only if there's valid audio data
            if np.any(audio_data):  # Check if audio data contains non-zero values
                reduced_noise_data = nr.reduce_noise(y=audio_data, sr=self.sample_rate)

                # Calculate RMS value, ensuring no invalid data (NaN) is used
                if np.any(reduced_noise_data):  # Check for valid noise-reduced data
                    rms_value = np.sqrt(np.mean(np.square(reduced_noise_data)))

                    # Only add frames that are below the noise threshold (i.e., filter out ambient noise)
                    if not np.isnan(rms_value) and rms_value < self.noise_threshold:
                        frames.append(reduced_noise_data.astype(int16).tobytes())
                else:
                    print("Invalid reduced noise data encountered.")
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

            # Collect the transcription into the list
            for segment in segments:
                # print(f"Transcription: {segment.text}")
                transcribed_text.append(segment.text)

        if transcribed_text:
            return " ".join(transcribed_text)  # Return the transcribed text as a single string

            
    def dynamic_threshold(self, rms_values, factor=1.5):
        """Adjust noise threshold dynamically based on the median RMS."""
        median_rms = np.median(rms_values)
        return median_rms * factor
    
    def stream_output(self, text):
        import urllib.parse
        # Example parameters
        voice = "maxheadroom_00000045.wav"
        language = "en"
        output_file = "stream_output.wav"
        
        # Encode the text for URL
        encoded_text = urllib.parse.quote(text)
        
        # Create the streaming URL
        streaming_url = f"http://localhost:7851/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        
        # Create and play the audio stream using VLC
        player = vlc.MediaPlayer(streaming_url)
        
        def on_end_reached(event):
            print("End of stream reached.")
            player.stop()
        
        # Attach event to detect when the stream ends
        event_manager = player.event_manager()
        event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_end_reached)
        
        # Start playing the stream
        player.play()
        
        # Keep the script running to allow the stream to play
        while True:
            state = player.get_state()
            if state in [vlc.State.Ended, vlc.State.Stopped, vlc.State.Error]:
                break
            time.sleep(1)
          
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
            

# sp = Speak()
# sp.glitch_stream_output2("this is a test of pitch and stutter. test 1 2 3. I just need a long enough sentence to see the frequecy of sound changes.")

# print(sp.listen3())