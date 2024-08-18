import requests
import winsound

# Use the API endpoint to generate TTS
url = "http://127.0.0.1:7851/api/tts-generate"

# Prepare the form data
data = {
    "text_input": "This is a test stream.",
    "text_filtering": "standard",
    "character_voice_gen": "maxheadroom_00000005.wav",
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
response = requests.post(url, data=data)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response to get the file URL
    result = response.json()
    audio_url = result['output_file_url']
    
    # Download the audio file
    audio_response = requests.get(audio_url)
    
    # Save the audio file locally
    with open("output.wav", "wb") as f:
        f.write(audio_response.content)
    winsound.PlaySound('output.wav', winsound.SND_FILENAME)
else:
    print(f"Failed with status code {response.status_code}: {response.text}")
