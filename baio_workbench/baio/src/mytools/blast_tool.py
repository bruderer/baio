from typing import Optional
from pydantic import BaseModel, Field
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain
)
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents.agent_toolkits import create_python_agent
import os
from langchain.chat_models import ChatOpenAI
from baio.src.non_llm_tools.utilities import Utils, JSONUtils
from langchain.callbacks import get_openai_callback
from typing import Optional, Sequence

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    PromptTemplate,
)
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import faiss

import requests
import time

import re

import requests
from pydantic import ValidationError

import time
from pydantic import BaseModel, Field, validator


os.environ["OPENAI_API_KEY"] = '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'
embeddings = OpenAIEmbeddings()

model_name = 'gpt-4'
llm = ChatOpenAI(model_name=model_name, temperature=0)


ncbi_jin_db = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/faissdb", embeddings)


class BlastQueryRequest(BaseModel):
    url: str = Field(
        default="https://blast.ncbi.nlm.nih.gov/Blast.cgi",
        description="URL endpoint for the NCBI BLAST API."
    )
    cmd: str = Field(
        default="Put",
        description="Command to execute, 'Put' for submitting query, 'Get' for retrieving results."
    )
    program: str = Field(
        ...,
        description="BLAST program to use, e.g., 'blastn' for nucleotide BLAST."
    )
    database: str = Field(
        default="nt",
        description="Database to search, e.g., 'nt' for nucleotide database."
    )
    query: Optional[str] = Field(
        None,
        description="Nucleotide or protein sequence for the BLAST query."
    )
    format_type: Optional[str] = Field(
        default="Text",
        description="Format of the BLAST results, e.g., 'Text', 'XML'."
    )
    rid: Optional[str] = Field(
        None,
        description="Request ID for retrieving BLAST results."
    )
    other_params: Optional[dict] = Field(
        default={"email": "noah.bruderer@uib.no"},
        description="Other optional BLAST parameters, including user email."
    )
    max_hits: int = Field(
        default=5,
        description="Maximum number of hits to return in the BLAST results."
    )

class NCBIAPI:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        template_api_ncbi = """
        You have to provide the necessary information to answer this question 
        Question: {question}\n\n
        Based on the explanaitions and example questions below:\n
        {context}
        """
        self.ncbi_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template_api_ncbi)
    def query(self, question: str) -> str:
        ncbi_qa_chain= ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self.memory,
            retriever=ncbi_jin_db.as_retriever(), 
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.ncbi_CHAIN_PROMPT},
            verbose=True,
        )

        relevant_api_call_info = ncbi_qa_chain(question)
        return relevant_api_call_info


class AnswerExtractor:
    """Extract answer from """
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        template_api_ncbi = """
        You have to answer the question:{question} in a clear and as short as possible manner, be factual!\n\n
        Based on the information given here:\n
        {context}
        """
        self.ncbi_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template_api_ncbi)
    def query(self,  question: str, file_path:str) -> str:
        #first embedd file
        loader = TextLoader(file_path)
        documents = loader.load()
        #split
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        #embed
        doc_embeddings = FAISS.from_documents(docs, embeddings)
        ncbi_qa_chain= ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self.memory,
            retriever=doc_embeddings.as_retriever(), 
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.ncbi_CHAIN_PROMPT},
            verbose=True,
        )

        relevant_api_call_info = ncbi_qa_chain(question)
        return relevant_api_call_info
    

def api_query_generator(question: str):
    """ function executing:
    1: text retrieval from ncbi_doc
    2: structured data from (1) to generate BlastQueryRequest object
    """
    ncbi_api = NCBIAPI()


    relevant_api_call_info = ncbi_api.query(question)

# Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=BlastQueryRequest)

    # Prompt
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Run Retrieval and write strucutred output 

    _input = prompt.format_prompt(query=question)
    print(_input.to_string())
    model = OpenAI(temperature=0)
    #output is a json, annoying but we have to deal with it
    output = model(_input.to_string())
    print(output)
    parser.parse(output)

    try:
        query_request = BlastQueryRequest.parse_raw(output)
        # Now you can use query_request as an instance of BlastQueryRequest
    except ValidationError as e:
        print(f"Validation error: {e}")
        return 'Failed to write API query instructions'
        # Handle validation error
    return query_request


def submit_blast_query(request_data: BlastQueryRequest):
    # Prepare the data for the POST request
    data = {
        'CMD': request_data.cmd,
        'PROGRAM': request_data.program,
        'DATABASE': request_data.database,
        'QUERY': request_data.query,
        'FORMAT_TYPE': request_data.format_type
    }
    # Include any other_params if provided
    if request_data.other_params:
        data.update(request_data.other_params)

    # Make the API call
    response = requests.post(request_data.url, data=data)
    response.raise_for_status()

    # Extract RID from response
    match = re.search(r"RID = (\w+)", response.text)
    if match:
        return match.group(1)
    else:
        raise ValueError("RID not found in BLAST submission response.")


def fetch_and_save_blast_results(rid: str, save_path, wait_time: int = 15, max_attempts: int = 10000):
    base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    check_status_params = {
        'CMD': 'Get',
        'FORMAT_OBJECT': 'SearchInfo',
        'RID': rid
    }
    get_results_params = {
        'CMD': 'Get',
        'FORMAT_TYPE': 'Text',
        'RID': rid
    }

    # Check the status of the BLAST job
    for attempt in range(max_attempts):
        status_response = requests.get(base_url, params=check_status_params)
        status_response.raise_for_status()
        status_text = status_response.text

        if 'Status=WAITING' in status_text:
            print(f"{rid} results not ready, waiting...")
            time.sleep(wait_time)
        elif 'Status=FAILED' in status_text:
            raise Exception("BLAST query failed.")
        elif 'Status=UNKNOWN' in status_text:
            raise Exception("BLAST query expired or does not exist.")
        elif 'Status=READY' in status_text:
            if 'ThereAreHits=yes' in status_text:
                print("{frid} results are ready, retrieving and saving...")
                results_response = requests.get(base_url, params=get_results_params)
                results_response.raise_for_status()
                # Save the results to a file
                with open(f'{save_path}/BLAST_results_{rid}.txt', 'w') as file:
                    file.write(results_response.text)
                print(f'Results saved in BLAST_results_{rid}.txt')
                break
            else:
                print("No hits found.")
                break
        else:
            raise Exception("Unknown status.")

    if attempt == max_attempts - 1:
        raise TimeoutError("Maximum attempts reached. Results may not be ready.")

def blast_result_file_extractor(file_path, question):
    """Extracting the answer from blast result file"""

    #extract answer
    answer_extractor = AnswerExtractor()
    return answer_extractor.query(question, file_path)
