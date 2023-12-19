# This module is meant for initializing and getting the ChatOpenAI object.
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

llm = None
llm35 = None  # Declare llm35 at the global scope
embedding = None

def initialize_llm(model_name, openai_api_key):
    global llm
    global llm35  # Declare llm35 as global within the function
    global embedding
    if llm is None:  # Only initialize if not already done
        llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
        embedding = OpenAIEmbeddings()
    if llm35 is None:  # Initialize llm35 if not already done
        llm35 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106", openai_api_key=openai_api_key)
        embedding = OpenAIEmbeddings()

def get_llm():
    if llm is None:
        raise Exception("LLM has not been initialized")
    return llm



# You should not initialize llm35 here if it's not used by all components
# Consider initializing it where it's actually needed, or provide a similar
# initialization function as you did for 'llm'.

