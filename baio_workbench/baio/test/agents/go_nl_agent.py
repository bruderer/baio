import os

from langchain.chat_models import ChatOpenAI

from baio.src.agents import go_nl_agent


def test_go_nl_agent():
    # Setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)
    result = go_nl_agent("What is the official gene symbol for LMP10?", llm)
    print(result)


test_go_nl_agent()
