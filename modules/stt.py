import speech_recognition as sr
import pyttsx3 

class STT:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

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

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
        
        
# while True:
#     stt = STT()
#     text = stt.listen()
#     stt.speak(text)
#     del stt
#     print("Listening again...")   
