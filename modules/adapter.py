# import environ
from langchain_core.prompts import ChatPromptTemplate

# env = environ.Env()
# environ.Env.read_env()

class Adapter:
    def __init__(self, env):
        self.llm_text = env("LLM_TYPE")
        if self.llm_text.lower() == "openai":
            from langchain_openai import OpenAIEmbeddings, OpenAI
            from langchain_openai import ChatOpenAI
            self.llm = OpenAI(temperature=0, openai_api_key=env("OPENAI_API_KEY"))
            self.prompt = ChatPromptTemplate.from_template(
                "answer the following request: {topic}"
            )
            self.llm_chat = ChatOpenAI(
                temperature=0.3, model=env("OPENAI_MODEL"), openai_api_key=env("OPENAI_API_KEY")
            )
            self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        elif self.llm_text.lower() == "local":
            from langchain_community.llms import Ollama
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            from langchain_community.chat_models import ChatOllama
            llm_model = env("OLLAMA_MODEL")
            self.llm = Ollama(base_url=env("OLLAMA_URL"), model=llm_model)
            self.prompt = ChatPromptTemplate.from_template(
                "answer the following request: {topic}"
            )
            self.llm_chat = ChatOllama(
                base_url=env("OLLAMA_URL"), model=llm_model
            )
            model_name = "BAAI/bge-small-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            self.embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        elif self.llm_text.lower() == "hybrid":
            from langchain_openai import OpenAIEmbeddings, OpenAI
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            self.llm = OpenAI(temperature=0, openai_api_key=env("OPENAI_API_KEY"))
            model_name = "BAAI/bge-small-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            self.embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        else:
            raise ValueError("Invalid LLM")

    def chat(self, query):
        print(f"adaptor query: {query}")
        from langchain_core.output_parsers import StrOutputParser
        chain = self.prompt | self.llm_chat | StrOutputParser()
        result = chain.invoke({"topic": query})
        return result

    