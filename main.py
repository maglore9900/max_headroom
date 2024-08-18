from modules import adapter, speak


sp = speak.Speak()
ad = adapter.Adapter("openai")


while True:
    text = sp.listen()
    response = ad.chat(text)
    sp.max_headroom(response)
    
    print("Listening again...")