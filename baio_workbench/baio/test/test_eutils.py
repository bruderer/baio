import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from baio.src.mytools.eutils import eutils_tool


def test_eutils_tool():
    # Setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)
    embedding = OpenAIEmbeddings(api_key=openai_api_key)

    # Test question
    question = "What is the official gene symbol for LMP10?"
    print("Testing eutils_tool with question:")
    print(question)
    print("\n")

    # Call the eutils_tool function
    result = eutils_tool(question, llm, embedding)
    print("\nResult:")
    print(result)

    # Check if the result is a string (the answer)
    assert isinstance(result, str), "Result should be a string (the answer)"

    # Check if the result is not empty
    assert result.strip() != "", "Result should not be empty"

    # Check if the log file was created
    log_file_path = "./baio/data/output/eutils/results/log_file/eutils_log.json"
    assert os.path.exists(log_file_path), f"Log file {log_file_path} does not exist"

    # Check if at least one result file was created
    file_path = "./baio/data/output/eutils/results/files/"
    assert any(
        file.startswith("eutils_results_") for file in os.listdir(file_path)
    ), f"No result files found in {file_path}"

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_eutils_tool()
