import requests
import winsound
import speech_recognition as sr
import pyttsx3 
import os

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
            print("Listening...")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                print("You said: ", text)
                return text
            except sr.UnknownValueError:
                print("Sorry, I didn't get that.")
            except sr.RequestError as e:
                print("Sorry, I couldn't request results; {0}".format(e))