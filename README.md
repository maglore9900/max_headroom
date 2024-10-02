this is a personal project to create a voice directed digital assistant based on the max headroom character.

![alt text](https://www.cartoonbrew.com/wp-content/uploads/2013/05/maxheadroom_main-1280x600.jpg)

# SUMMARY

written in python, using langchain, langgraph, etc.

written to work on Windows. Agent and logic will run on linux but tools are currently windows only.

it currently will respond as an LLM like usual, but also has the following capabilities:

- can also control spotify
- can open applications on windows
- can change the focused window
- set timer
- coming soon:  journalling in markdown, with a save target for obsidian

this is a fun work in progress. if you want to use it and or develop for it be my guest. would love to have more tools designed.

Note:

1) this will work with openai or ollama models. you will need to set up the .env for that as well as spotify
2) this is designed to pull a custom voice from the alltalk project https://github.com/erew123/alltalk_tts, that is how I am cloning max headroom's voice. You can alter or simply not use this, it will currently fallback to pyttsx3 aka a robot voice.
3) speech-to-text can use google, or faster-whisper. faster-whisper is currently the default and optimal method.

# INSTALLATION

so basically the steps are pretty simple

- download the code (clone it or download it and unzip it)
- install python 3.10 on the system
- create a virtual environment using `python -m venv .` in the folder/dir of the code
- activate the environment with `Scripts\activate.bat` on windows or `source bin/activate` on linux
- run pip install to install all the required modules `pip install -r requirements.txt`
- then copy example_env.txt to `.env`
- open that, and put in your info, like openai key or ollama or whatever
- then run `python main.py` to start the whole thing up

# TOOLS

## Spotify

you will need get your spotify credentials in order to have Max control your spotify software.

you can find information on getting that information here: https://developer.spotify.com/documentation/web-api/concepts/apps

max can take the following commands: play, pause, stop, next, previous, favorite

***note: you can say really any words that are similiar, max will attempt to read your intent and use the right command**

## Window Focus

this tool brings the focus of whatever app you name to the front, it will not open an app

***note: only works on windows**

## Open App

this tool will open an application. when you run max it will create an index of the apps installed on your system

***note: only works on windows**

## Timer

this tool will set a timer with a popup. you tell max to set a time for X time, it will convert it to seconds on the backend and create the timer.

the default timer will have a "clippy" popup, with potentially custom text
