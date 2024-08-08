import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from baio.src.mytools.aniseed import aniseed_tool

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)
embedding = OpenAIEmbeddings(api_key=openai_api_key)
ANISEED_db = FAISS.load_local(
    "./baio/data/persistant_files/vectorstores/aniseed", embedding
)


def test_aniseed_tool():
    # Test question
    question = (
        "What genes are expressed in Ciona intestinalis between stage 1 and 10"
        "in the notochord?"
    )

    print("Testing aniseed_tool with question:")
    print(question)
    print("\n")

    # Call the aniseed_tool function
    result = aniseed_tool(question, llm)

    print("\nResult:")
    print(result)

    # Check if the result is a list of file paths
    assert isinstance(result, list), "Result should be a list of file paths"

    # Check if all paths in the result exist
    for path in result:
        assert os.path.exists(path), f"File {path} does not exist"

    print("\nTest completed successfully!")


test_aniseed_tool()
