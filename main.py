from modules import adapter, speak, spotify


sp = speak.Speak()
ad = adapter.Adapter("openai")
spot = spotify.Spotify()


while True:
    text = sp.listen()
    if text and "max" in text.lower():
        response = ad.chat(text)
        
        # sp.max_headroom(response)
        sp.glitch_stream_output(response)
    
    print("Listening again...")