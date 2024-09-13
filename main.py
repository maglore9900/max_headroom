from modules import agent
import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
graph = agent.Agent()

while True:
    text = graph.spk.listen()
    # if text:
        # print(f"User: {text}")
    if text and "hey" in text.lower() and "max" in text.lower() or text and "hey" in text.lower() and "mac" in text.lower():
        if "exit" in text.lower():
            break
        print("agent invoked")
        response = loop.run_until_complete(graph.invoke_agent(text))
        if response:
            graph.spk.glitch_stream_output(response)
    
