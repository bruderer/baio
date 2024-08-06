import json
import os
import tempfile

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


class BLASTAnswerExtractor:
    """Extract answer from BLAST result files"""

    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        BLAST_file_answer_extractor_prompt = """
        You have to answer the question:{question} as clear and short as possible manner
        , be factual!\n\
        If you are asked what organism a specific sequence belongs to check in the
        'Hit_def' fields, if you find a synthetic construc or predicted etc. move to the
        next entery and look of an orgnaism name\n\
        Try to use the hits with the best identity score to answer the question, if it
        is not possible move to the next one. \n\
        Be clear, and if organism names are present in ANY of the result please use them
        in the answer, do not make up stuff and mention how relevant the found
        information is (based on the identity scores)
        Based on the information given here:\n\
        {context}
        """
        self.BLAST_file_answer_extractor_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=BLAST_file_answer_extractor_prompt,
        )

    def query(self, question: str, file_path: str, n: int, embedding, llm) -> dict:
        # we make a short of the top hits of the files
        first_n_lines = []
        with open(file_path, "r") as file:
            for _ in range(n):
                line = file.readline()
                if not line:
                    break
                first_n_lines.append(line)
        # Create a temporary file and write the lines to it
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file.writelines(first_n_lines)
            temp_file_path = temp_file.name
        if os.path.exists(temp_file_path):
            print(temp_file_path)
            loader = TextLoader(temp_file_path)
        else:
            raise FileNotFoundError(f"Temporary file not found: {temp_file_path}")
        # loader = TextLoader(temp_file_path)
        documents = loader.load()
        os.remove(temp_file_path)
        # split
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        # embed
        doc_embeddings = FAISS.from_documents(docs, embedding)
        BLAST_answer_extraction_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self.memory,
            retriever=doc_embeddings.as_retriever(),
            return_source_documents=False,
            combine_docs_chain_kwargs={
                "prompt": self.BLAST_file_answer_extractor_prompt
            },
            verbose=True,
        )
        BLAST_answer = BLAST_answer_extraction_chain(question)
        return BLAST_answer


def BLAST_answer(log_file_path, question, current_uuid, n_lignes: int, embedding, llm):
    print("in Answer function:")
    with open(log_file_path, "r") as file:
        data = json.load(file)
    print(current_uuid)
    # Access the last entry in the JSON array
    last_entry = data[-1]
    # Extract the file path
    current_file_path = last_entry["file_path"]
    print("3: Extracting answer")
    answer_extractor = BLASTAnswerExtractor()
    result = answer_extractor.query(
        question, current_file_path, n_lignes, embedding, llm
    )
    print(result)
    for entry in data:
        if entry["uuid"] == current_uuid:
            entry["answer"] = result["answer"]
            break
    with open(log_file_path, "w") as file:
        json.dump(data, file, indent=4)
    return result
