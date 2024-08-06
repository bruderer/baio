# ./baio/src/llm/__init__.py

# This module is meant for initializing and getting the ChatOpenAI object.

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


class LLM:
    _instance_35 = None
    _instance_selected = None
    _embedding = None

    @staticmethod
    def initialize(openai_api_key, selected_model):
        LLM._instance_35 = LLM._instance_35 or ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key
        )
        LLM._instance_selected = LLM._instance_selected or ChatOpenAI(
            model_name=selected_model, temperature=0, openai_api_key=openai_api_key
        )
        LLM._embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    @staticmethod
    def get_instance(model_name=None):
        if model_name == "gpt-3.5-turbo":
            return LLM._instance_35
        elif model_name is None:
            return LLM._instance_selected
        else:
            raise ValueError("Invalid model name")

    @staticmethod
    def get_embedding():
        if LLM._embedding is None:
            raise Exception("Embedding has not been initialized")
        return LLM._embedding
