import json
import os
import tempfile

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


class AnswerExtractor:
    """Extract answer for BLATresults"""

    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        template_api_eutils = """
        You have to answer the question:{question} as clear and short as possible, be
        factual! Be precise and if there are mismatches, say so!\n\
        Example question: Align the DNA sequence to the human genome:
        ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGT\
        GGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT"\
        Output to find answer in: "track": "blat", "genome": "hg38", "fields":\
        ["matches", "misMatches", "repMatches", "nCount", "qNumInsert", "qBaseInsert",\
        "tNumInsert", "tBaseInsert", "strand", "qName", "qSize", "qStart", "qEnd",\
        "tName", "tSize", "tStart", "tEnd", "blockCount", "blockSizes", "qStarts",\
        "tStarts"], "blat": [[128, 0, 0, 0, 0, 0, 0, 0, "+", "YourSeq", 128, 0, 128,\
        "chr15", 101991189, 91950804, 91950932, 1, "128", "0", "91950804"], [31, 0, 0,\
        0, 1, 54, 1, 73, "-", "YourSeq", 128, 33, 118, "chr6", 170805979, 48013377,\
        48013481, 2, "14,17", "10,78", "48013377,48013464"], [29, 0, 0, 0, 0, 0, 1, 114\
        ,"-", "YourSeq", 128, 89, 118, "chr9", 138394717, 125385023, 125385166, 2, "13,\
        16", "10,23", "125385023,125385150"], [26, 1, 0, 0, 0, 0, 1, 2, "+", "YourSeq",\
        128, 1, 28, "chr17", 83257441, 62760282, 62760311, 2, "5,22", "1,6", "62760282,\
        62760289"], [24, 3, 0, 0, 0, 0, 0, 0, "-", "YourSeq", 128, 54, 81,\
        "chr11_KI270832v1_alt", 210133, 136044, 136071, 1, "27", "47", "136044"], [20,0\
        , 0, 0, 0, 0, 0, 0, "+", "YourSeq", 128, 106, 126, "chr2", 242193529, 99136832,\
        99136852, 1, "20", "106", "99136832"]]\
        Find the tStart and tEnd fields, be sure to use the exact same integers as in
        the output. Always use the best matches, do not give more than 3 examples if
        there are multiple matches\
        Answer: chr15:91950804-91950932; Identity score: 100%; Matches: 128\n\
        Based on the information given below \n\
        {context}
        """
        self.eutils_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], template=template_api_eutils
        )

    def query(self, question: str, file_path: str, llm, embedding) -> dict:
        # we make a short extract of the top hits of the files
        first_400_lines = []
        with open(file_path, "r") as file:
            for _ in range(400):
                line = file.readline()
                if not line:
                    break
                first_400_lines.append(line)
        # Create a temporary file and write the lines to it
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file.writelines(first_400_lines)
            temp_file_path = temp_file.name
        if os.path.exists(temp_file_path):
            loader = TextLoader(temp_file_path)
        else:
            print(f"Temporary file not found: {temp_file_path}")
        # loader = TextLoader(temp_file_path)
        documents = loader.load()
        os.remove(temp_file_path)
        # split
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        # embed
        doc_embeddings = FAISS.from_documents(docs, embedding)
        ncbi_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self.memory,
            retriever=doc_embeddings.as_retriever(),
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.eutils_CHAIN_PROMPT},
            verbose=True,
        )
        relevant_api_call_info = ncbi_qa_chain(question)
        return relevant_api_call_info


def BLAT_answer(current_file_path, question, llm, embedding):
    # with open(log_file_path, "r") as file:
    #     data = json.load(file)
    # print(current_uuid)
    # # Access the last entry in the JSON array
    # last_entry = data[-1]
    # # Extract the file path
    # current_file_path = last_entry["file_path"]
    print("3: Extracting answer")
    answer_extractor = AnswerExtractor()
    result = answer_extractor.query(question, current_file_path, llm, embedding)
    print(result)
    # for entry in data:
    #     if entry["uuid"] == current_uuid:
    #         entry["answer"] = result["answer"]
    #         with open(log_file_path, "w") as file:
    #             json.dump(data, file, indent=4)
    return result["answer"]
