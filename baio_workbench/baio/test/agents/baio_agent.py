import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from baio.src.agents.baio_agent import baio_agent  # Adjust the import path as needed


def test_baio_agent():
    # Setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)
    embedding = OpenAIEmbeddings(api_key=openai_api_key)

    # Test questions
    questions = [
        "What is the official gene symbol for LMP10?",
        "What is the protein sequence for the gene BRCA1?",
        "Align the DNA sequence ATCGATCGATCGATCG to the human genome.",
    ]

    for question in questions:
        print("\nTesting baio_agent with question:")
        print(question)
        print("\n")

        # Call the baio_agent function
        result = baio_agent(question, llm, embedding)

        print("\nResult:")
        print(result)

        # Check if the result is a string (the answer)
        assert isinstance(result, str), "Result should be a string (the answer)"

        # Check if the result is not empty
        assert result.strip() != "", "Result should not be empty"

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_baio_agent()
