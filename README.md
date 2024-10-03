this is a personal project to create a voice directed digital assistant based on the max headroom character.

![alt text](https://www.cartoonbrew.com/wp-content/uploads/2013/05/maxheadroom_main-1280x600.jpg)

# SUMMARY

written in python, using langchain, langgraph, etc.

written to work on Windows. Agent and logic will run on linux but some tools are currently windows only.

it currently will respond as an LLM like usual, but also has the following capabilities:

- custom prompt options
- can also control spotify
- can open applications on windows
- can change the focused window
- set timer
- coming soon:  journalling in markdown, with a save target for obsidian

this is a fun work in progress. if you want to use it and or develop for it be my guest. would love to have more tools designed.

Note:

1) this will work with openai or ollama models. you will need to set up the .env for that as well as spotify
2) this is designed to pull a custom voice from the [alltalk project https://github.com/erew123/alltalk_tts, that is how I am cloning max headroom's voice. You can alter or simply not use this, it will currently fallback to pyttsx3 aka a robot voice.
3) speech-to-text can use google, or faster-whisper. faster-whisper is currently the default and optimal method.

# INSTALLATION

so basically the steps are pretty simple

- download the code (clone it or download it and unzip it)
- install python 3.10 on the system
- create a virtual environment using `python -m venv .` in the folder/dir of the code
- activate the environment with `Scripts\activate.bat` on windows or `source bin/activate` on linux
- run pip install to install all the required modules `pip install -r requirements_windows.txt`
- then copy example_env.txt to `.env`
- open that, and put in your info, like openai key or ollama or whatever
- then run `python main.py` to start the whole thing up

# TOOLS

## Spotify

you will need get your spotify credentials in order to have Max control your spotify software.

you can find information on getting that information here: https://developer.spotify.com/documentation/web-api/concepts/apps

max can take the following commands: play, pause, stop, next, previous, favorite

`hey max play spotify` for example

***note: you can say really any words that are similiar, max will attempt to read your intent and use the right command**

## Window Focus

this tool brings the focus of whatever app you name to the front, it will not open an app

`hey max show obisidian` for example

***note: only works on windows**

## Open App

this tool will open an application. when you run max it will create an index of the apps installed on your system

`hey max open obsidian` for example

***note: only works on windows**

## Timer

this tool will set a timer with a popup. you tell max to set a time for X time, it will convert it to seconds on the backend and create the timer.

the default timer will have a "clippy" popup, with potentially custom text

`hey max set timer 2 hours` for example

# Customization

## Prompt

Max Headroom is the default prompt. If you want to make a custom prompt look in modules/prompts.py and add it there. then set the name in .env

## Alert Phrase/Wake Word

Max is set up for "Hey Max" as the wake word. I didnt love "hey max" as opposed to just "max" but the number of times he got invoked randomly became rediculous.

If you want to modify the wake word look in main.py, you will see the logic where it looks at the speech to text detected and looks for the key words. you can make this whatever you want

## Speech

Max has a unique stutter glitch and I recreated this by modifying the voice stream as its being received. If you want to use all-talk with a different model or just dont want glitchiness then comment out `graph.spk.glitch_stream_output(response)` in main.py and uncomment `graph.spk.stream(response)`.

for a custom voice selection look at `modules/speak.py` under the function `stream` and set the voice model there. I will probably make this easier, aka in the .env at some point.

# Process Flow

![Alt text](images/flow.png)
