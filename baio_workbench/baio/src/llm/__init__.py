# This module is meant for initializing and getting the ChatOpenAI object.
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

llm = None

def initialize_llm(model_name, openai_api_key):
    global llm
    if llm is None:  # Only initialize if not already done
        llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)

def get_llm():
    if llm is None:
        raise Exception("LLM has not been initialized")
    return llm

import os
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = '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'
# embeddings = OpenAIEmbeddings()

model_name = 'gpt-4'
llm = ChatOpenAI(model_name=model_name, temperature=0)
llm35 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
embeddings = OpenAIEmbeddings()

# You should not initialize llm35 here if it's not used by all components
# Consider initializing it where it's actually needed, or provide a similar
# initialization function as you did for 'llm'.

