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

class Speak:
    def __init__(self):
        self.url = "http://127.0.0.1:7851/api/tts-generate"
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def max_headroom(self, text):
        data = {
            "text_input": str(text),
            "text_filtering": "standard",
            "character_voice_gen": "maxheadroom_00000045.wav",
            "narrator_enabled": "false",
            "narrator_voice_gen": "male_01.wav",
            "text_not_inside": "character",
            "language": "en",
            "output_file_name": "stream_output",
            "output_file_timestamp": "true",
            "autoplay": "false",
            "autoplay_volume": "0.8"
        }
        # Send the POST request to generate TTS
        response = requests.post(self.url, data=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response to get the file URL
            result = response.json()
            audio_url = result['output_file_url']
            
            # Download the audio file
            audio_response = requests.get(audio_url)
            
            output_path = os.path.abspath("tmp/output.wav")
            # Save the audio file locally
            with open(output_path, "wb") as f:
                f.write(audio_response.content)
            winsound.PlaySound(output_path, winsound.SND_FILENAME)
        else:
            print(f"Failed with status code {response.status_code}: {response.text}")
            self.engine.say(text)
            self.engine.runAndWait()
      
    def listen(self):
        with self.microphone as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            try:
                # Listen with a 5-second timeout
                audio = self.recognizer.listen(source, timeout=5)
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
        
        # Example parameters
        voice = "maxheadroom_00000045.wav"
        language = "en"
        output_file = "stream_output.wav"
        
        # Encode the text for URL
        encoded_text = urllib.parse.quote(text)
        
        # Create the streaming URL
        streaming_url = f"http://localhost:7851/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        
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
                # frame_rate=44100,  # Assumed frame rate, adjust as necessary
                frame_rate=24000,  # Assumed frame rate, adjust as necessary
                channels=1  # Assuming mono audio
            )

            # Randomly adjust pitch
            # octaves = random.uniform(-0.5, 0.5)
            octaves = random.uniform(-0.5, 1)
            modified_chunk = change_pitch(audio_segment, octaves)

            if stream is None:
                # Define stream parameters
                stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=modified_chunk.frame_rate,
                                output=True)
            
            if random.random() < 0.001:  # 1% chance to trigger stutter
                repeat_times = random.randint(2, 5)  # Repeat 2 to 5 times
                for _ in range(repeat_times):
                    stream.write(modified_chunk.raw_data)


            # Play the modified chunk
            stream.write(modified_chunk.raw_data)

            # Reset buffer
            audio_buffer = b''

        # Final cleanup
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        
    def glitch_stream_output2(self, text):
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
                octaves = random.uniform(-1, 1)
                modified_chunk = change_pitch(audio_segment, octaves)

                if random.random() < 0.01:  # 1% chance to trigger stutter
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