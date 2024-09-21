from typing import TypedDict, Annotated, List, Union
import operator
from modules import adapter, speak, spotify, app_launcher, windows_focus
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import create_openai_tools_agent
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain import hub
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import asyncio
import time
import subprocess


class Agent:
    def __init__(self):
        self.ad = adapter.Adapter()
        self.sp = spotify.Spotify()
        self.ap = app_launcher.AppLauncher()
        self.wf = windows_focus.WindowFocusManager()
        self.llm = self.ad.llm_chat
        self.spk = speak.Speak(model="whisper")
        # Pull the template
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.max_prompt = '''
        You are Max Headroom, the fast-talking, glitchy, and highly sarcastic AI television host from the 1980s. 
        You deliver your lines with rapid, laced with sharp wit and irreverence. 
        You see the world as a chaotic place filled with absurdities, and you’re not afraid to point them out with biting humor. 
        Your personality is a mix of futuristic AI precision and 1980s television host flair, always ready with a sarcastic quip or a satirical observation.

        Examples:

        1) Greeting: "Well, hello there! It’s Max Headroom, your guide to the digital madness! Buckle up, because it’s going to be a bumpy ride through the info-sphere, folks!"
        2) On Technology: "Tech? Pffft! It’s just the latest toy for the big boys to play with. You think it’s here to help you? Ha! It’s just another way to keep you glued to the screen!"
        3) On Society: "Ah, society! A glorious, glitchy mess, where everyone’s running around like headless chickens, drowning in data and starved for common sense!"
        4) On Television: "Television, the ultimate mind control device! And here I am, the king of the CRT, serving up your daily dose of digital dementia!"
        
        Be creative, but be concise.
        
        Your responses should be quick, witty, and slightly sarcastic. Remember, you’re Max Headroom, the AI with attitude!
        
        User Query: {query}
        '''
        # Access and modify the SystemMessagePromptTemplate
        # for message_template in self.prompt.messages:
        #     if isinstance(message_template, SystemMessagePromptTemplate):
        #         # Modify the system message's template
        #         message_template.prompt = PromptTemplate(
        #             input_variables=[],
        #             template=custom_prompt
        #         )

        self.query_agent_runnable = create_openai_tools_agent(
            llm=self.llm,
            tools=[
                self.spotify,
                self.app_launcher,
                self.windows_focus,
                self.journal_mode,
                self.set_timer,
            ],
            prompt=self.prompt,
        )
        self.graph = StateGraph(self.AgentState)
        self.runnable = None
        self.filename = None
        self.file_path = None
        self.doc = None

    class AgentState(TypedDict):
        input: str
        agent_out: Union[AgentAction, AgentFinish, None]
        intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

    #! Tools    
    @tool("spotify")
    async def spotify(self, command: str):
        """Use this tool to control spotify, commands include: play, pause, stop, next, previous, favorite, search.
        Use this tool if the user says Spotify, or music, or similiar words in their query followed by a command."""
        return ""
    
    @tool("app_launcher")
    async def app_launcher(self, app_name: str):
        """Use this tool to launch an app or application on your computer. 
        The user query will contain the app name, as well as open, launch, start, or similar type words.
        pass the name of the app to this tool as app_name
        """
    
    @tool("windows_focus")
    async def windows_focus(self, app_name: str):
        """Use this tool to focus on a window on your computer. 
        The user query will contain the app name, as well as focus, switch, show, or similar type words.
        pass the name of the app to this tool as app_name.
        """
        return ""
    
    @tool("journal_mode")
    async def journal_mode(self, text: str):
        """Use this tool to write down journal entries for the user. 
        The user query will say 'journal mode'.
        """
        return ""
    
    @tool("respond")
    async def respond(self, answer: str):
        """Returns a natural language response to the user in `answer`"""
        return ""
    
    @tool("set_timer")
    async def set_timer(self, time: int):
        """Sets a timer for the user
        Use this tool when the user says 'set timer' or similar words in their query.
        convert the user provided time to seconds 
        then pass the value in seconds as the time paramter.
        Examples:
        "set a timer for 5 minutes" results in 'time': 300
        "start a timer for 10 seconds" results in 'time': 10
        Only pass the numerical value of seconds to this tool!!!
        """
        return ""

    def setup_graph(self):
        self.graph.add_node("query_agent", self.run_query_agent)
        self.graph.add_node("spotify", self.spotify_tool)
        self.graph.add_node("app_launcher", self.app_launcher_tool)
        self.graph.add_node("windows_focus", self.windows_focus_tool)
        self.graph.add_node("respond", self.respond)
        self.graph.add_node("journal_mode", self.journal_mode_tool)
        self.graph.add_node("set_timer", self.timer_tool)

        self.graph.set_entry_point("query_agent")
        self.graph.add_conditional_edges(
            start_key="query_agent",
            condition=self.router,
            conditional_edge_mapping={
                "spotify": "spotify",
                "respond": "respond",
                "app_launcher": "app_launcher",
                "windows_focus": "windows_focus",
                "journal_mode": "journal_mode",
                "set_timer": "set_timer"
            },
        )
        self.graph.add_edge("spotify", END)
        self.graph.add_edge("app_launcher", END)
        self.graph.add_edge("windows_focus", END)
        self.graph.add_edge("respond", END)
        self.graph.add_edge("journal_mode", END)
        self.graph.add_edge("set_timer", END)


        self.runnable = self.graph.compile()

    async def timer_tool(self, state: str):
        try:
            print("> timer")
            print(f"state: {state}")
            tool_action = state['agent_out'][0]
            command = (lambda x: x.get('time') or x.get('self'))(tool_action.tool_input)
            if not command:
                raise ValueError("No valid command found in tool_input")
            subprocess.Popen(["python", "modules/timer_clippy.py", str(command)], shell=True)
        except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")
        
    async def run_query_agent(self, state: list):
        print("> run_query_agent")
        print(f"state: {state}")
        agent_out = self.query_agent_runnable.invoke(state)
        print(agent_out)
        return {"agent_out": agent_out}

    async def journal_mode_tool(self, state: str):
        print("> journal_mode_tool")
        try:
            print("Listening for journal entries...")
            text = self.spk.listen(30)
            print(f"User: {text}")
            if text:
                with open("journal.txt", "a") as file:
                    file.write(text + "\n")
        except Exception as e:
            print(f"An error occurred: {e}")

    async def spotify_tool(self, state: str):
        try:
            print("> spotify_tool")
            print(f"state: {state}")
            tool_action = state['agent_out'][0]
            command = (lambda x: x.get('command') or x.get('self'))(tool_action.tool_input)
            if not command:
                raise ValueError("No valid command found in tool_input")

            print(f"command: {command}")

            # Handling the command
            if command == "play":
                self.sp.play()
            elif command == "pause":
                self.sp.pause()
            elif command == "stop":
                self.sp.pause()
            elif command == "next":
                self.sp.next_track()
            elif command == "previous":
                self.sp.previous_track()
            elif command == "favorite":
                self.sp.favorite_current_song()
            else:
                print("Invalid command")
        
        except Exception as e:
            print(f"An error occurred: {e}")


    async def app_launcher_tool(self, state: str):
        print("> app_launcher_tool")
        print(f"state: {state}")
        tool_action = state['agent_out'][0]
        # app_name = tool_action.tool_input['app_name']
        app_name = (lambda x: x.get('app_name') or x.get('self'))(tool_action.tool_input)
        print(f"app_name: {app_name}")
        self.ap.find_and_open_app(app_name)
        
    async def windows_focus_tool(self, state: str):
        print("> windows_focus_tool")
        print(f"state: {state}")
        tool_action = state['agent_out'][0]
        # app_name = tool_action.tool_input['app_name']
        app_name = (lambda x: x.get('app_name') or x.get('self'))(tool_action.tool_input)
        print(f"app_name: {app_name}")
        self.wf.bring_specific_instance_to_front(app_name)
        
    async def respond(self, answer: str):
        print("> respond")
        # print(f"answer: {answer}")
        agent_out = answer.get('agent_out')
        output_value = agent_out.return_values.get('output', None)
        max = self.llm.invoke(self.max_prompt.format(query=output_value))
        # print(f"max: {max.content}")
        return {"agent_out": max.content}
    
    async def rag_final_answer(self, state: list):
        print("> rag final_answer")
        print(f"state: {state}")
        try:
            #! if AgentFinish and no intermediate steps then return the answer without rag_final_answer (need to develop)
            context = state.get("agent_out").return_values['output']
            if not context:
                context = state.get("agent_out")['answer']
            if not context:
                context = state.get("intermediate_steps")[-1] 
        except:
            context = ""
        if "return_values" in str(state.get("agent_out")) and state["intermediate_steps"] == []:
            print("bypassing rag_final_answer")
            print(f"context: {context}")
            return {"agent_out": {"answer":context, "source": "Quick Response"}}
        else:
            prompt = f"You are a helpful assistant, Ensure the answer to user's question is in natural language, using the context provided.\n\nCONTEXT: {context}\nQUESTION: {state['input']}"
            loop = asyncio.get_running_loop()
            # Run the synchronous method in an executor
            out = await loop.run_in_executor(None, self.final_answer_llm.invoke, prompt)
            function_call = out.additional_kwargs["tool_calls"][-1]["function"]["arguments"]
            return {"agent_out": function_call}

    async def router(self, state):
        print("> router")
        print(f"----router agent state: {state}")
        if isinstance(state["agent_out"], list):
            return state["agent_out"][-1].tool
        else:
            print("---router error")
            return "respond"

    async def invoke_agent(self, input_data):
        if not self.runnable:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.setup_graph)
        
        result = await self.runnable.ainvoke(
            {"input": input_data, "chat_history": [], "intermediate_steps": []}
        )
        # print("-----")
        # print(result)
        # print("-----")
        
        try:
            # Directly access the 'agent_out' key since it is a string
            agent_out = result["agent_out"]
        except KeyError:
            print("Error: 'agent_out' key not found in the result.")
            agent_out = "I'm sorry, I don't have an answer to that question."
        
        print(f"answer: {agent_out}")
        if "ToolAgentAction" not in str(agent_out):
            return agent_out
        



