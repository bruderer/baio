import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from baio.src.mytools.blat import BLAT_tool


def test_BLAT_tool():
    # Setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)
    embedding = OpenAIEmbeddings(api_key=openai_api_key)

    # Load the vector store

    # Test question
    question = (
        "Align the DNA sequence to the human genome:ACACAGTAAAGATCATTTATTTATAAAC"
        "TGACAATGGCTGCTTTAACACTAGTAGAGCAGAGTCAAGTAATTGTGACAAAGACCAACCGGCCCATAAATTTGAAAA"
        "TTTGTACTCTCTAGTCCTTTGTGTGAGAAGTTTGCTAA"
    )
    print("Testing BLAT_tool with question:")
    print(question)
    print("\n")

    # Call the BLAT_tool function
    result = BLAT_tool(question, llm, embedding)

    print("\nResult:")
    print(result)

    # Check if the result is a string (the answer)
    assert isinstance(result, str), "Result should be a string (the answer)"

    # Check if the result is not empty
    assert result.strip() != "", "Result should not be empty"

    # Check if the log file was created
    log_file_path = "./baio/data/output/BLAT/logfile.json"
    assert os.path.exists(log_file_path), f"Log file {log_file_path} does not exist"

    # Check if at least one result file was created
    file_path = "./baio/data/output/BLAT/files/"
    assert any(
        os.path.isfile(os.path.join(file_path, file)) for file in os.listdir(file_path)
    ), f"No result files found in {file_path}"

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_BLAT_tool()
