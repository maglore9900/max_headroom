from modules import agent
import asyncio
import environ
import os

env = environ.Env()
environ.Env.read_env()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


if os.name == "nt":
    op = "windows"
elif os.name == "posix":
    # Further check to differentiate between Linux and macOS
    if 'linux' in os.uname().sysname.lower():
        print("linux")
        op = "linux"
    elif 'darwin' in os.uname().sysname.lower():
        op = "macos"
    else:
        exit("Unknown operating system.")
else:
    exit("Unknown operating system.")

graph = agent.Agent(env,op)

while True:
    text = graph.spk.listen()
    # if text:
        # print(f"User: {text}")
    if text and "hey" in text.lower() and env("CHARACTER") in text.lower():
        if "exit" in text.lower():
            break
        print("agent invoked")
        response = loop.run_until_complete(graph.invoke_agent(text))
        if response:
            graph.spk.glitch_stream_output(response)
            # graph.spk.stream(response)
    
