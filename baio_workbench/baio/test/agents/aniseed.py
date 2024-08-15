import os

from langchain.chat_models import ChatOpenAI

from baio.src.agents.aniseed_agent import (
    aniseed_agent,  # Adjust the import path as needed
)


def test_baio_agent():
    # Setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)

    # Test questions
    questions = [
        (
            "What genes are expressed in ciona intestinalis between stage 1 and 10? "
            "and in what anatomical territory?"
        )
    ]

    for question in questions:
        print("\nTesting baio_agent with question:")
        print(question)
        print("\n")

        # Call the baio_agent function
        result = aniseed_agent(question, llm)

        print("\nResult:")
        print(result)

        # Check if the result is a string (the answer)
        assert isinstance(result, list), "Result should be a string (the answer)"

        # Check if the result is not empty

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_baio_agent()
