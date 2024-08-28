from typing import TypedDict, Annotated, List, Union
import operator
from modules import adapter, spotify, app_launcher
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import create_openai_tools_agent
from langchain import hub
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import asyncio




class Agent:
    def __init__(self):
        self.ad = adapter.Adapter()
        self.sp = spotify.Spotify()
        self.ap = app_launcher.AppLauncher()
        self.llm = self.ad.llm_chat
        # self.final_answer_llm = self.llm.bind_tools(
        #     [self.rag_final_answer_tool], tool_choice="rag_final_answer"
        # )

        self.prompt = hub.pull("hwchase17/openai-functions-agent")

        self.query_agent_runnable = create_openai_tools_agent(
            llm=self.llm,
            tools=[
                # self.rag_final_answer_tool,
                self.spotify,
                self.app_launcher
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
    @tool("respond")
    async def respond(self, answer: str):
        """Returns a natural language response to the user in `answer`"""
        return ""
    
    @tool("spotify")
    async def spotify(self, command: str):
        """Use this tool to control spotify, commands include: play, pause, next, previous, favorite, search
        Only use this tool if the user says Spotify in their query"""
        return ""
    
    @tool("app_launcher")
    async def app_launcher(self, app_name: str):
        """Use this tool to launch an app or application on your computer. 
        The user query will contain the app name, as well as open, launch, start, or similar type words
        pass the name of the app to this tool as app_name
        """
    
    # @tool("rag_final_answer")
    # async def rag_final_answer_tool(self, answer: str, source: str):
    #     """Returns a natural language response to the user in `answer`, and a
    #     `source` which provides citations for where this information came from.
    #     """
    #     return ""
    
    

    def setup_graph(self):
        self.graph.add_node("query_agent", self.run_query_agent)
        self.graph.add_node("spotify", self.spotify_tool)
        self.graph.add_node("app_launcher", self.app_launcher_tool)
        # self.graph.add_node("rag_final_answer", self.rag_final_answer)
        # self.graph.add_node("error", self.rag_final_answer)
        self.graph.add_node("respond", self.respond)

        self.graph.set_entry_point("query_agent")
        self.graph.add_conditional_edges(
            start_key="query_agent",
            condition=self.router,
            conditional_edge_mapping={
                "spotify": "spotify",
                # "rag_final_answer": "rag_final_answer",
                # "error": "error",
                "respond": "respond",
                "app_launcher": "app_launcher",
            },
        )
        self.graph.add_edge("spotify", END)
        self.graph.add_edge("app_launcher", END)
        # self.graph.add_edge("error", END)
        # self.graph.add_edge("rag_final_answer", END)
        # self.graph.add_edge("query_agent", END)
        self.graph.add_edge("respond", END)


        self.runnable = self.graph.compile()

    async def run_query_agent(self, state: list):
        print("> run_query_agent")
        print(f"state: {state}")
        agent_out = self.query_agent_runnable.invoke(state)
        print(agent_out)
        return {"agent_out": agent_out}

    async def spotify_tool(self, state: str):
        print("> spotify_tool")
        print(f"state: {state}")
        tool_action = state['agent_out'][0]
        command = tool_action.tool_input['command']
        print(f"command: {command}")
        # print(f"search: {search}")
        if command == "play":
            self.sp.play()
        elif command == "pause":
            self.sp.pause()
        elif command == "next":
            self.sp.next_track()
        elif command == "previous":
            self.sp.previous_track()
        elif command == "favorite":
            self.sp.favorite_current_song()
        elif command == "search":
            self.sp.search_song_and_play(search)
        else:
            print("Invalid command")

    async def app_launcher_tool(self, state: str):
        print("> app_launcher_tool")
        print(f"state: {state}")
        tool_action = state['agent_out'][0]
        app_name = tool_action.tool_input['app_name']
        print(f"app_name: {app_name}")
        # print(f"search: {search}")
        self.ap.find_and_open_app(app_name)
        
    async def respond(self, answer: str):
        print("> respond")
        print(f"answer: {answer}")
        # answer = answer.agent_out.return_values.get('output', None)
        agent_out = answer.get('agent_out')
        output_value = agent_out.return_values.get('output', None)
        return {"agent_out": output_value}
    
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
        print("-----")
        print(result)
        print("-----")
        
        try:
            # Directly access the 'agent_out' key since it is a string
            agent_out = result["agent_out"]
        except KeyError:
            print("Error: 'agent_out' key not found in the result.")
            agent_out = "I'm sorry, I don't have an answer to that question."
        
        # 'agent_out' is already the answer in this case
        answer = agent_out
        
        print(f"answer: {answer}")
        if "ToolAgentAction" not in str(agent_out):
            return answer



