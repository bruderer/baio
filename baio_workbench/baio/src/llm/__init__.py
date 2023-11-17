# This module is meant for initializing and getting the ChatOpenAI object.

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

# You should not initialize llm35 here if it's not used by all components
# Consider initializing it where it's actually needed, or provide a similar
# initialization function as you did for 'llm'.

