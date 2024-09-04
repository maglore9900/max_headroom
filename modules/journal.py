from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
import adapter

ad = adapter.Adapter()

class Journal:
    def __init__(self):
        self.sys_prompt = """
            You are tasked with creating a detailed and neutral journal entry. The entry should capture key points, observations, any relevant metrics, and recommended actions or next steps. The language should remain neutral, avoiding any self-referential language.

            Example 1:
            Date: [DD-MM-YYYY]

            **Key Points:**
            - [Summary of the main points or events, capturing essential details]

            **Observations:**
            - [Notes on observations, trends, or insights drawn from the data]

            **Metrics:**
            - [List of any relevant metrics, figures, or statistics related to the data]

            **Recommended Actions:**
            - [Suggested actions, strategies, or next steps based on the data analysis]


            Example 2:
            Date: [DD-MM-YYYY]

            **Summary:**
            - [Overview of the data or event, focusing on significant highlights]

            **Analysis:**
            - [Detailed analysis, including patterns, anomalies, or key insights]

            **Figures:**
            - [Relevant numerical data, metrics, or charts]

            **Next Steps:**
            - [Proposed actions, decisions, or follow-up activities derived from the analysis]

            Ensure that the tone remains neutral, and avoid using 'I' or 'my' in the notes.
            
            The format MUST be in markdown
            """
    
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        self.sys_prompt
                    )
                ),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )
        
        self.journal_llm = self.chat_template | ad.llm_chat | StrOutputParser()
        
    def journal(self, text):
        message = self.chat_template.format_messages(text=text)
        response = self.journal_llm.invoke(message)
        return response


