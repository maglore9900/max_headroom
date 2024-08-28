from modules import agent, speak
import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


sp = speak.Speak()
graph = agent.Agent()


while True:
    text = sp.listen2()
    if text and "max" in text.lower():
        if "exit" in text.lower():
            break
        response = loop.run_until_complete(graph.invoke_agent(text))
        if response:
            sp.glitch_stream_output2(response)
    
