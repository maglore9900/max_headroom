from modules import adapter, stt

stt = stt.STT()
ad = adapter.Adapter("openai")


while True:
    text = stt.listen()
    response = ad.chat(text)
    stt.speak(response)
    print("Listening again...")