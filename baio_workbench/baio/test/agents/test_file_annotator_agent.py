import os

from langchain.chat_models import ChatOpenAI

from baio.src.agents import csv_chatter_agent


def test_csv_chatter_agent():
    # Setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)
    path = ["./baio/test/data/aniseed_dummy_out.csv"]
    question = "whats up with this file?"
    result = csv_chatter_agent(question, path, llm)
    print(result)


test_csv_chatter_agent()
