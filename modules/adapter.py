from pathlib import Path
import environ
import os
# import psycopg2
from typing import Dict, List, Optional, Tuple, Annotated

# import pandas as pd  # Uncomment this if you need pandas

# langchain imports
# from langchain.agents import AgentExecutor, tool, create_openai_functions_agent
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain 
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    AzureAIDocumentIntelligenceLoader,
)

# from langchain.sql_database import SQLDatabase

from langchain_community.utilities import SQLDatabase
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.sql import SQLDatabaseChain

# from langchain_openai import ChatOpenAI

# sqlalchemy imports
from sqlalchemy import create_engine

from langsmith import traceable

import asyncio
import re
import math
import json

from time import time


env = environ.Env()
environ.Env.read_env()


class Adapter:
    def __init__(self, llm_type):
        self.llm_text = llm_type
        #! IP and Credentials for DB
        # self.engine = create_engine(f"postgresql+psycopg2://postgres:{env('DBPASS')}@10.0.0.141:9999/{env('DBNAME')}")
        # self.engine = create_engine(
        #     f"postgresql+psycopg2://postgres:{env('DBPASS')}@10.0.0.141:9999/{env('DBNAME')}"
        # )
        # #! max string length
        # self.db = SQLDatabase(engine=self.engine, max_string_length=1024)
        # self.db_params = {
        #     "dbname": env("DBNAME"),
        #     "user": "postgres",
        #     "password": env("DBPASS"),
        #     "host": "10.0.0.141",  # or your database host
        #     "port": "9999",  # or your database port
        # }
        # self.conn = psycopg2.connect(**self.db_params)
        # self.cursor = self.conn.cursor()

        if self.llm_text.lower() == "openai":
            from langchain_openai import OpenAIEmbeddings, OpenAI
            from langchain_openai import ChatOpenAI

            self.llm = OpenAI(temperature=0, openai_api_key=env("OPENAI_API_KEY"))
            self.prompt = ChatPromptTemplate.from_template(
                "answer the following request: {topic}"
            )
            self.llm_chat = ChatOpenAI(
                temperature=0.3, openai_api_key=env("OPENAI_API_KEY")
            )
            self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        elif self.llm_text.lower() == "azure":
            from langchain_openai import AzureChatOpenAI
            from langchain_openai import AzureOpenAIEmbeddings

            # self.llm = AzureChatOpenAI(azure_deployment=env("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"), openai_api_version=env("AZURE_OPENAI_API_VERSION"))
            self.llm_chat = AzureChatOpenAI(
                temperature=0,
                azure_deployment=env("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                openai_api_version=env("AZURE_OPENAI_API_VERSION"),
                response_format={"type": "json_object"},
            )
            self.embedding = AzureOpenAIEmbeddings(
                temperature=0,
                azure_deployment=env("AZURE_OPENAI_CHAT_DEPLOYMENT_EMBED_NAME"),
                openai_api_version=env("AZURE_OPENAI_API_VERSION"),
            )
        elif self.llm_text.lower() == "local":
            from langchain_community.llms import Ollama
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            from langchain_community.chat_models import ChatOllama

            llm_model = "llama3"
            # llm_model = "notus"
            self.llm = Ollama(base_url="http://10.0.0.231:11434", model=llm_model)
            self.prompt = ChatPromptTemplate.from_template(
                "answer the following request: {topic}"
            )
            self.llm_chat = ChatOllama(
                base_url="http://10.0.0.231:11434", model=llm_model
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

    def load_document(self, filename):
        file_path = "uploads/" + filename
        # Map file extensions to their corresponding loader classes

        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".md": UnstructuredMarkdownLoader,
            ".odt": UnstructuredODTLoader,
            ".ppt": UnstructuredPowerPointLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".xlsx": UnstructuredExcelLoader,
        }

        # Identify the loader based on file extension
        for extension, loader_cls in loaders.items():
            if filename.endswith(extension):
                loader = loader_cls(file_path)
                documents = loader.load()
                break
        else:
            # If no loader is found for the file extension
            raise ValueError("Invalid file type")
        # print(f"documents: {documents}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30
        )
        return text_splitter.split_documents(documents=documents)
   
    def add_to_datastore(self):
        try:
            filename = input("Enter the name of the document (.pdf or .txt):\n")
            docs = self.load_document(filename)
            #! permanent vector store
            datastore_name = os.path.splitext(filename) + "_datastore"
            vectorstore = FAISS.from_documents(docs, self.embedding)
            vectorstore.save_local(datastore_name)
        except Exception as e:
            print(e)

    def add_many_to_datastore(self, src_path, dest_path=None):
        start = time()
        vectorstore = None
        count = 0
        if not dest_path:
            dest_path = src_path
        datastore_name = dest_path + "_datastore"
        entries = os.listdir(src_path)
        # print(entries)
        files = [
            entry for entry in entries if os.path.isfile(os.path.join(src_path, entry))
        ]
        for each in files:
            try:
                # print(each)
                doc = self.load_document(f"{src_path}/{each}")
                # print(doc)
                if not Path(datastore_name).exists():
                    vectorstore = FAISS.from_documents(doc, self.embedding)
                    vectorstore.save_local(datastore_name)
                else:
                    if vectorstore is None:
                        vectorstore = FAISS.load_local(datastore_name, self.embedding)
                    tmp_vectorstore = FAISS.from_documents(doc, self.embedding)
                    vectorstore.merge_from(tmp_vectorstore)
                    vectorstore.save_local(datastore_name)
                count += 1
                print(count)
            except Exception as e:
                print(e)
        end = time()
        print(end - start)

    def query_datastore(self, query, datastore):
        try:
            retriever = FAISS.load_local(datastore, self.embedding).as_retriever()
            qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=retriever, verbose=True
            )
            # qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, verbose=True)
            if self.llm_text.lower() == "openai" or self.llm_text.lower() == "hybrid":
                # result = qa.invoke(query)['result']
                result = qa.invoke(query)
            else:
                result = qa.invoke(query)
            return result
        except Exception as e:
            print(e)
            
    def agent_query_doc(self, query, doc):
        qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=doc, verbose=True
            )
        result = qa.invoke(query)
        return result
    
    def vector_doc(self, filename):
        doc = self.load_document(filename)
        retriever = FAISS.from_documents(doc, self.embedding).as_retriever()
        # retriever = self.hybrid_retrievers(doc, "doc")
        return retriever
    
    def query_doc(self, query, filename, doc):
        # from langchain_community.vectorstores import Qdrant

        try:
            print(f"query: {query}")
            print(f"filename: {filename}")
            # doc = self.load_document(filename, file_path)

            #! permanent vector store
            # print(f"here is the document data {doc}")
            # vectorstore = FAISS.from_documents(docs, self.embedding)
            # vectorstore.save_local("faiss_index_constitution")
            # persisted_vectorstore = FAISS.load_local("faiss_index_constitution", self.embedding)
            #! impermanent vector store
            retriever = FAISS.from_documents(doc, self.embedding).as_retriever()
            # retriever = self.hybrid_retrievers(doc)
            #! qdrant options instead of FAISS, need to explore more metadata options for sources
            # qdrant = Qdrant.from_documents(
            #     doc,
            #     self.embedding,
            #     location=":memory:",  # Local mode with in-memory storage only
            #     collection_name="my_documents",
            # )
            # retriever = qdrant.as_retriever()
            # qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, verbose=True)
            qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=retriever, verbose=True
            )
            query = qa.invoke(query)
            # result = query['answer']
            # source = query['sources']
            # return result+"\nSource:"+source
            return query
        except Exception as e:
            print(e)

    def query_db(self, query):
        """Answer all Risk Management Framework (RMF) control and CCI related questions."""

        QUERY = """
        Given an input question, first create a syntactically correct postgresql query to run, then look at all of the results of the query. Return an answer for all matches.
        
        When returning an answer always format the response like this. 
        RMF Control: <rmf_control>
        CCI: <rmf_control_cci>
        Assessment Procedurse: <assessment_procedures for rmf_control_cci>
        Implementation Guidance: <implementation_guidance for rmf_control_cci>
        
        
        DO NOT LIMIT the length of the SQL query or the response.
        {question}
        """

        db_chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose=True)
        try:
            question = QUERY.format(question=query)
            if self.llm_text.lower() == "openai" or self.llm_text.lower() == "hybrid":
                result = str(db_chain.invoke(question)["result"])
            else:
                result = db_chain.invoke(question)
            return result
        except Exception as e:
            print(e)

    def compare(self, query, db):
        try:
            docs = self.load_document("test.txt")
            for each in docs:
                print(each.page_content)
                response = self.query_db(f"\n{query} {each.page_content}\n", db)
                return response
        except Exception as e:
            print(e)

    # def tokenize(self, data):
    #     #! chunk and vector raw data
    #     try:
    #         for each in list(data):
    #             print(each)
    #         # results = self.embedding.embed_query(data)
    #         # print(results[:5])
    #     except Exception as e:
    #         print(e)

    #! modified with formatter
    def chain_query_db(self, prompt):
        #! use LLM to translate user question into SQL query
        QUERY = """
        Given an input question, create a syntactically correct postgresql query to run. Do not limit the return DO NOT USE UNION. DO NOT LIMIT the length of the SQL query or the response. Do NOT assume RMF control number or any other data types.
        {question}
        """
        db_query = SQLDatabaseChain.from_llm(
            self.llm, self.db, verbose=False, return_sql=True
        )
        try:
            question = QUERY.format(question=prompt)
            if self.llm_text.lower() == "openai" or self.llm_text.lower() == "hybrid":
                result = db_query.invoke(question)["result"]
            else:
                result = db_query.invoke(question)
            # print(f"this is the result query: {result}")
            self.cursor.execute(f"{result};")
            db_data = self.cursor.fetchall()
            db_data = sorted(db_data)
            print(f"-------- db_data: {db_data}\n")
            formated = self.query_db_format_response(result, db_data)
            print(f"formated response: {formated}")
            # return(db_data)
            return formated
        except Exception as e:
            print(e)

    #! new helper function
    def query_extractor(self, sql_query):
        # Split the query at 'UNION', if 'UNION' is not present, this will simply take the entire query
        parts = sql_query.split(" UNION ")
        column_names = []

        # Only process the last part after the last 'UNION'
        if len(parts) > 1:
            part = parts[-1]  # This gets the last segment after the UNION
        else:
            part = parts[
                0
            ]  # This handles cases without any UNION, taking the whole query

        # Extract the text between 'SELECT' and 'FROM'
        selected_part = part.split("SELECT")[1].split("FROM")[0].strip()
        # Split the selected part on commas to get individual column names
        columns = [column.strip() for column in selected_part.split(",")]
        # Remove table aliases and extra quotes if present
        for column in columns:
            # Remove table prefix if exists (e.g., table_name.column_name)
            if "." in column:
                column = column.split(".")[-1]
            # Strip quotes and whitespaces around the column names
            clean_column = column.strip().strip('"').strip()
            # Append all columns to the list, allowing duplicates
            column_names.append(clean_column)

        return column_names

    #! response formatter
    def query_db_format_response(self, sql_query, response):
        sql_query_list = self.query_extractor(sql_query)
        print(f"sql response: {response}")
        print(f"SQL Query List: {sql_query_list}")
        columns = sql_query_list
        data_dict = {}
        control_list = [
            "rmf_control_number",
            "rmf_control_family",
            "rmf_control_title",
            "rmf_control_text",
            "confidentiality",
            "integrity",
            "availability",
            "supplementary_guidance",
            "criticality",
        ]
        cci_list = [
            "rmf_control_cci",
            "rmf_control_cci_def",
            "implementation_guidance",
            "assessment_procedures",
            "confidentiality",
            "integrity",
            "availability",
        ]

        for record in response:
            record_dict = {column: record[idx] for idx, column in enumerate(columns)}
            rmf_control_number = record_dict.get("rmf_control_text_indicator")

            print(f"rmf_control_text_indicator: {rmf_control_number}")
            # print(f"record: {record}")
            if not rmf_control_number:
                rmf_control_number = record_dict.get("rmf_control_number")
                print(f"rmf_control_number: {rmf_control_number}")
            else:
                match = re.search(r"rmf_control_number\s*=\s*\'([^\']*)\'", sql_query)
                if match:
                    rmf_control_number = match.group(1)
                    print(f"rmf_control_group: {rmf_control_number}")
            rmf_control_cci = record_dict.pop("rmf_control_cci", None)

            if rmf_control_number:
                # Ensure a dictionary exists for this control number
                if rmf_control_number not in data_dict:
                    data_dict[rmf_control_number] = {"CCI": {}}

                # Handle CCI values specifically
                if rmf_control_cci:
                    # Ensure a dictionary exists for this CCI under the control number
                    if rmf_control_cci not in data_dict[rmf_control_number]["CCI"]:
                        data_dict[rmf_control_number]["CCI"][rmf_control_cci] = {}

                    # Populate the CCI dictionary with relevant data from record_dict
                    for key in record_dict:
                        if key in cci_list:
                            # Initialize or append to the list for each key
                            if (
                                key
                                not in data_dict[rmf_control_number]["CCI"][
                                    rmf_control_cci
                                ]
                            ):
                                data_dict[rmf_control_number]["CCI"][rmf_control_cci][
                                    key
                                ] = []
                            value = record_dict[key]
                            if isinstance(value, float) and math.isnan(value):
                                value = None
                            data_dict[rmf_control_number]["CCI"][rmf_control_cci][
                                key
                            ].append(record_dict[key])

                for key in record_dict:
                    if key in control_list:
                        if key not in data_dict[rmf_control_number]:
                            data_dict[rmf_control_number][key] = []
                        value = record_dict[key]
                        if isinstance(value, float) and math.isnan(value):
                            value = None
                        if value not in data_dict[rmf_control_number][key]:
                            data_dict[rmf_control_number][key].append(value)
                response = json.dumps(data_dict, indent=4)
            else:
                response = [list(item) for item in response]
        print(f"response: {response}")
        # json_output = json.dumps(data_dict, indent=4)
        # return json_output
        return response

    def chat(self, query):
        print(f"adaptor query: {query}")
        from langchain_core.output_parsers import StrOutputParser

        chain = self.prompt | self.llm_chat | StrOutputParser()
        # loop = asyncio.get_running_loop()
        # Run the synchronous method in an executor
        # result = await loop.run_in_executor(None, chain.invoke({"topic": query}))
        result = chain.invoke({"topic": query})
        # print(f"adapter result: {result}")
        return result

    #! multi-doc loader with one output, attempted to dev for general purpose, may not need it for other purposes
    def multi_doc_loader(self, files: Annotated[list, "List of files to load"]):
        print("multi_doc_loader")
        docs = []
        for file in files:
            doc = self.load_document(file)
            docs.extend(doc)
        return docs
    
    #! helper function needs to have multi_doc_loader to be run first and that value to be docs
    def multi_doc_splitter(self, docs):
        print("multi_doc_splitter")
        from langchain import hub
        from langchain_core.runnables import RunnablePassthrough
        d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )
        chunk_size_tok = 2000
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size_tok, chunk_overlap=0
        )
        texts_split = text_splitter.split_text(concatenated_content)
        return texts_split
    
    def raptorize(self, docs):
        texts = self.multi_doc_loader(docs)
        texts_split = self.multi_doc_splitter(texts)
        import raptor
        from langchain import hub
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        rapt = raptor.Raptor(self.llm_chat, self.embedding)
        raptor_results = rapt.recursive_embed_cluster_summarize(texts_split, level=1, n_levels=3)
        print("raptor run")
        for level in sorted(raptor_results.keys()):
            # Extract summaries from the current level's DataFrame
            summaries = raptor_results[level][1]["summaries"].tolist()
            # Extend all_texts with the summaries from the current level
            texts_split.extend(summaries)
        # vectorstore = FAISS.from_texts(texts_split, self.embedding)
        # retriever = vectorstore.as_retriever()
        retriever = self.hybrid_retrievers(texts_split, "text")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm_chat
            | StrOutputParser()
        )

        ccis = """The organization conducting the inspection/assessment examines the information system to ensure the organization being inspected/assessed configures the information system to audit the execution of privileged functions.
        """
        # For information system components that have applicable STIGs or SRGs, the organization conducting the inspection/assessment evaluates the components to ensure that the organization being inspected/assessed has configured the information system in compliance with the applicable STIGs and SRGs pertaining to CCI 2234."""

        # Question
        print(rag_chain.invoke(f"search the document for any information that best satisifies the following Question: {ccis}. \n make sure you quote the section of the document where the information was found."))

    def hybrid_retrievers(self, doc, type):
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import FAISS
        if type.lower() == "text":
            bm25_retriever = BM25Retriever.from_texts(
                doc, metadatas=[{"source": 1}] * len(doc)
                )
            bm25_retriever.k = 2
            faiss_vectorstore = FAISS.from_texts(
                doc, self.embedding, metadatas=[{"source": 2}] * len(doc)
            )
        elif type.lower() == "doc":
            bm25_retriever = BM25Retriever.from_documents(doc)
            faiss_vectorstore = FAISS.from_documents(doc, self.embedding)
            
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        return ensemble_retriever
    

    ############
    
    
    def vector_doc2(self, doc, retriever_type, weight=None):
        if "hybrid" in retriever_type.lower():
            if "faiss" in retriever_type.lower():
                retriever = self.hybrid_retrievers2(doc, "doc", "faiss", weight)
            elif "qdrant" in retriever_type.lower():
                retriever = self.hybrid_retrievers2(doc, "doc", "qdrant", weight)
        elif "faiss" in retriever_type.lower():
            retriever = FAISS.from_documents(doc, self.embedding).as_retriever()
        elif "chroma" in retriever_type.lower():
            from langchain_chroma import Chroma
            retriever = Chroma.from_documents(doc, self.embedding).as_retriever()
        elif "qdrant" in retriever_type.lower():
            from langchain_community.vectorstores import Qdrant
            qdrant = Qdrant.from_documents(
                doc,
                self.embedding,
                location=":memory:",  # Local mode with in-memory storage only
                # collection_name="my_documents",
            )
            retriever = qdrant.as_retriever()

        return retriever
    
    def hybrid_retrievers2(self, doc, ret_type, doc_type, weight):
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.vectorstores import FAISS
        if "text" in doc_type.lower():
            bm25_retriever = BM25Retriever.from_texts(
                doc, metadatas=[{"source": 1}] * len(doc)
                )
            bm25_retriever.k = 2
            if "faiss" in ret_type.lower():
                vectorstore = FAISS.from_texts(
                    doc, self.embedding, metadatas=[{"source": 2}] * len(doc)
                )
            elif "qdrant" in ret_type.lower():
                from langchain_community.vectorstores import Qdrant
                qdrant = Qdrant.from_texts(
                    doc,
                    self.embedding,
                    location=":memory:",  # Local mode with in-memory storage only
                    # collection_name="my_documents",
                )
                vectorstore = qdrant
        elif "doc" in doc_type.lower():
            bm25_retriever = BM25Retriever.from_documents(doc)
            if "faiss" in ret_type.lower():
                vectorstore = FAISS.from_documents(doc, self.embedding)
            elif "qdrant" in ret_type.lower():
                from langchain_community.vectorstores import Qdrant
                qdrant = Qdrant.from_documents(
                    doc,
                    self.embedding,
                    location=":memory:",  # Local mode with in-memory storage only
                    # collection_name="my_documents",
                )
                vectorstore = qdrant
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[(1.0-float(weight)), float(weight)]
        )
        return ensemble_retriever
    
    # def raptorize2(self, query, docs, retriever_type, filename, weight=0.5):
    #     texts = self.multi_doc_loader(docs)
    #     texts_split = self.multi_doc_splitter(texts)
    #     import raptor
    #     from langchain import hub
    #     from langchain_core.runnables import RunnablePassthrough
    #     from langchain_core.output_parsers import StrOutputParser
    #     rapt = raptor.Raptor(self.llm_chat, self.embedding)
    #     raptor_results = rapt.recursive_embed_cluster_summarize(texts_split, level=1, n_levels=3)
    #     print("raptor run")
    #     for level in sorted(raptor_results.keys()):
    #         # Extract summaries from the current level's DataFrame
    #         summaries = raptor_results[level][1]["summaries"].tolist()
    #         # Extend all_texts with the summaries from the current level
    #         texts_split.extend(summaries)
    #     # vectorstore = FAISS.from_texts(texts_split, self.embedding)
    #     # retriever = vectorstore.as_retriever()
        
    #     if "faiss" in retriever_type.lower():
    #         #! chain requires source, this is a hack, does not add source
    #         # retriever = FAISS.from_texts(texts_split, self.embedding, metadatas=[{"source": 2}] * len(texts_split)).as_retriever()
    #         retriever = FAISS.from_texts(texts_split, self.embedding).as_retriever()
    #     elif "chroma" in retriever_type.lower():
    #         from langchain_chroma import Chroma
    #         retriever = Chroma.from_texts(texts_split, self.embedding).as_retriever()
    #     elif "qdrant" in retriever_type.lower():
    #         from langchain_community.vectorstores import Qdrant
    #         qdrant = Qdrant.from_texts(
    #             texts_split,
    #             self.embedding,
    #             location=":memory:",  # Local mode with in-memory storage only
    #             # collection_name="my_documents",
    #         )
    #         retriever = qdrant.as_retriever()
    #     elif "hybrid" in retriever_type.lower():
    #         if "faiss" in retriever_type.lower():
    #             retriever = self.hybrid_retrievers2(texts_split, "faiss", "text", weight)
    #         elif "qdrant" in retriever_type.lower():
    #             retriever = self.hybrid_retrievers2(texts_split, "qdrant", "text", weight)
        
    #     def format_docs(docs):
    #         return "\n\n".join(doc.page_content for doc in docs)
    #     #! creates multiple queries based on the first
    #     # retriever = MultiQueryRetriever.from_llm(
    #     #     llm=self.llm, retriever=retriever
    #     # )
        
    #     #! need to find actual source for this to have value
    #     # qa = RetrievalQAWithSourcesChain.from_chain_type(
    #     #         llm=self.llm, chain_type="stuff", retriever=retriever, verbose=False
    #     #     )
        
            
    #     prompt = hub.pull("rlm/rag-prompt")
    #     rag_chain = (
    #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #         | prompt
    #         | self.llm_chat
    #         | StrOutputParser()
    #     )
        
    #     import time
    #     start_time = time.perf_counter()
    #     result = rag_chain.invoke(query)
    #     # result = qa.invoke(query)
    #     end_time = time.perf_counter()
    #     total_time = end_time - start_time
    #     return result, total_time
    
    def raptorize2(self, query, docs, retriever_type, filename, weight=None):
        from langchain.schema import Document
        texts = self.multi_doc_loader(docs)
        texts_split = self.multi_doc_splitter(texts)
        import raptor
        from langchain import hub
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        rapt = raptor.Raptor(self.llm_chat, self.embedding)
        raptor_results = rapt.recursive_embed_cluster_summarize(texts_split, level=1, n_levels=3)
        print("raptor run")
        for level in sorted(raptor_results.keys()):
            # Extract summaries from the current level's DataFrame
            summaries = raptor_results[level][1]["summaries"].tolist()
            # Extend all_texts with the summaries from the current level
            texts_split.extend(summaries)
        # vectorstore = FAISS.from_texts(texts_split, self.embedding)
        # retriever = vectorstore.as_retriever()
        modified_list = []
        for each in texts_split:
            doc = Document(page_content=each, metadata={'source': filename})
            modified_list.append(doc)
        if weight is not None:
            if "doc" in filename.lower():
                vectorstore = self.hybrid_retrievers2(modified_list, retriever_type, "doc", weight)
            else:
                vectorstore = self.hybrid_retrievers2(modified_list, retriever_type, "text", weight)
        else:
            vectorstore = self.vector_doc2(modified_list, retriever_type)

        
       
        #! creates multiple queries based on the first
        # retriever = MultiQueryRetriever.from_llm(
        #     llm=self.llm, retriever=retriever
        # )
        
        #! need to find actual source for this to have value
        qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=vectorstore, verbose=False
            )
        
        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content for doc in docs)
            
        # prompt = hub.pull("rlm/rag-prompt")
        # rag_chain = (
        #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #     | prompt
        #     | self.llm_chat
        #     | StrOutputParser()
        # )
        
        import time
        start_time = time.perf_counter()
        # result = rag_chain.invoke(query)
        result = qa.invoke(query)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return result, total_time
        
    def agent_query_doc2(self, query, doc):
        # doc = MultiQueryRetriever.from_llm(
        #     llm=self.llm, retriever=doc
        # )
        qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=doc, verbose=False
            )
        import time
        start_time = time.perf_counter()
        result = qa.invoke(query)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return result, total_time
    
    def chroma_test(self, query, docs):
        from langchain_chroma import Chroma
        retriever = Chroma.from_documents(docs, self.embedding).as_retriever()
        retriever.invoke(query)
        
    def adj_sentence_clustering(self, text):
        import numpy as np
        import spacy
        nlp = spacy.load('en_core_web_sm')
        def process(text):
            doc = nlp(text)
            sents = list(doc.sents)
            vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

            return sents, vecs

        def cluster_text(sents, vecs, threshold):
            clusters = [[0]]
            for i in range(1, len(sents)):
                if np.dot(vecs[i], vecs[i-1]) < threshold:
                    clusters.append([])
                clusters[-1].append(i)
            
            return clusters

        def clean_text(text):
            # Add your text cleaning process here
            return text

        # Initialize the clusters lengths list and final texts list
        clusters_lens = []
        final_texts = []

        # Process the chunk
        threshold = 0.3
        sents, vecs = process(text)

        # Cluster the sentences
        clusters = cluster_text(sents, vecs, threshold)

        for cluster in clusters:
            cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))
            cluster_len = len(cluster_txt)
            
            # Check if the cluster is too short
            if cluster_len < 60:
                continue
            
            # Check if the cluster is too long
            elif cluster_len > 3000:
                threshold = 0.6
                sents_div, vecs_div = process(cluster_txt)
                reclusters = cluster_text(sents_div, vecs_div, threshold)
                
                for subcluster in reclusters:
                    div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
                    div_len = len(div_txt)
                    
                    if div_len < 60 or div_len > 3000:
                        continue
                    
                    clusters_lens.append(div_len)
                    final_texts.append(div_txt)
                    
            else:
                clusters_lens.append(cluster_len)
                final_texts.append(cluster_txt)
        return final_texts
    
    def load_document2(self, filename):
        from langchain.schema import Document
        file_path = "uploads/" + filename
        # Map file extensions to their corresponding loader classes

        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".md": UnstructuredMarkdownLoader,
            ".odt": UnstructuredODTLoader,
            ".ppt": UnstructuredPowerPointLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".xlsx": UnstructuredExcelLoader,
        }

        # Identify the loader based on file extension
        for extension, loader_cls in loaders.items():
            if filename.endswith(extension):
                loader = loader_cls(file_path)
                documents = loader.load()
                break
        else:
            # If no loader is found for the file extension
            raise ValueError("Invalid file type")
        
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000, chunk_overlap=30
        # )
        # result = text_splitter.split_documents(documents=documents)

        text = "".join(doc.page_content for doc in documents)

        cluster = self.adj_sentence_clustering(text)

        modified_list = []
        for each in cluster:
            doc = Document(page_content=each, metadata={'source': filename})
            modified_list.append(doc)
        # vectorstore = FAISS.from_documents(modified_list, self.embedding).as_retriever()
        # return vectorstore
        return modified_list
        