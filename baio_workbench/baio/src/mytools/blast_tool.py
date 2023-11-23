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
import uuid
import requests
import time

import re
from urllib.parse import urlencode
import requests
from pydantic import ValidationError
import json
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
        description="URL endpoint for API calls. 1: the NCBI BLAST API. 2: DNA alignnment to human genome use: 'https://genome.ucsc.edu/cgi-bin/hgBlat'?"
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
        default=15,
        description="Maximum number of hits to return in the BLAST results."
    )
    sort_by: Optional[str] = Field(
        default="score",
        description="Criterion to sort BLAST results by, e.g., 'score', 'evalue'."
    )
    megablast: str = Field(
        default="on", 
        description="Set to 'on' for human genome alignemnts"
    )
    # Additional fields for UCSC Genome Browser
    ucsc_genome: str = Field(
        default="hg38",
        description="Genome assembly to use in the UCSC Genome Browser."
    )
    ucsc_track: str = Field(
        default="genes",
        description="Genome Browser track to use, e.g., 'genes', 'gcPercent'."
    )
    ucsc_region: Optional[str] = Field(
        None,
        description="Region of interest in the genome, e.g., 'chr1:100000-200000'."
    )
    ucsc_output_format: str = Field(
        default="json",
        description="Output format for the UCSC Genome Browser, e.g., 'bed', 'fasta'."
    )
    ucsc_other_params: Optional[dict] = Field(
        default={},
        description="Other optional parameters for the UCSC Genome Browser."
    )
    ucsc_query_type:str = Field(
        default='DNA',
        description='depends on the query DNA, protein, translated RNA, or translated DNA'
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question."
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
        You have to answer the question:{question} as clear and short as possible manner, be factual!\n\
        For any kind of BLAST results use try to use the hit with the best idenity score to answer the questin, if it is not possible move to the next one. \n\
        If you are asked for gene alignments, use the nomencalture as followd: ChrN:start-STOP with N being the number of the chromosome.\n\
        The numbers before and after the hyphen indicate the start and end positions, respectively, in base pairs. This range is inclusive, meaning it includes both the start and end positions.\n\
        Based on the information given here:\n\
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
    #handle the question: either for ncbi or uscs
    # Prepare the data for the POST request 
    if 'ucsc' in request_data.url:
        #for uscs the request is onlu written here and submitted in the next function
        data = {
            'userSeq' : request_data.query,
            'type': request_data.ucsc_query_type,
            'db': request_data.ucsc_genome,
            'output': 'json'
        }
        # Make the API call
        query_string = urlencode(data)
        # Combine base URL with the query string
        full_url = f"{request_data.url}{query_string}"
        return full_url
    if 'blast' in request_data.url:
        data = {
            'CMD': request_data.cmd,
            'PROGRAM': request_data.program,
            'DATABASE': request_data.database,
            'QUERY': request_data.query,
            'FORMAT_TYPE': request_data.format_type,
            'MEGABLAST':request_data.megablast,
            'HITLIST_SIZE':request_data.max_hits,
        }
        # Include any other_params if provided
        if request_data.other_params:
            data.update(request_data.other_params)
        # Make the API call
        query_string = urlencode(data)
        # Combine base URL with the query string
        full_url = f"{request_data.url}?{query_string}"
        # Print the full URL
        print("Full URL for the request:", full_url)
        response = requests.post(request_data.url, data=data)
        response.raise_for_status()
        # Extract RID from response
        match = re.search(r"RID = (\w+)", response.text)
        if match:
            return match.group(1)
        else:
            raise ValueError("RID not found in BLAST submission response.")

def log_question_uuid(question_uuid, question, log_file_path):
    directory = os.path.dirname(log_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Read existing data or initialize new data
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Add new entry
    data.append({"uuid": question_uuid, "question": question})

    # Write updated data
    with open(log_file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
def fetch_and_save_blast_results(request_data: BlastQueryRequest, blast_query_return: str, save_path: str , question: str, log_file_path: str, wait_time: int = 15, max_attempts: int = 10000):
    request_data.question_uuid=str(uuid.uuid4())
    log_question_uuid(request_data.question_uuid, question, log_file_path)
    if 'ucsc' in request_data.url:
        response = requests.post(blast_query_return)
        response.raise_for_status()
        if response.status_code == 200:
            blat_result = response.json()       
            with open(f'{save_path}/BLAT_results_{request_data.question_uuid}.txt', 'w') as f:
                json.dump(blat_result, f, indent=4)
            return blat_result
        else:
            blat_result = f"Error: {response.status_code}"
            return blat_result
    if 'blast' in request_data.url:
        base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
        check_status_params = {
            'CMD': 'Get',
            'FORMAT_OBJECT': 'SearchInfo',
            'RID': blast_query_return
        }
        get_results_params = {
            'CMD': 'Get',
            'FORMAT_TYPE': 'XML',
            'RID': blast_query_return
        }
        # Check the status of the BLAST job
        for attempt in range(max_attempts):
            status_response = requests.get(base_url, params=check_status_params)
            status_response.raise_for_status()
            status_text = status_response.text
            if 'Status=WAITING' in status_text:
                print(f"{request_data.question_uuid} results not ready, waiting...")
                time.sleep(wait_time)
            elif 'Status=FAILED' in status_text:
                raise Exception("BLAST query failed.")
            elif 'Status=UNKNOWN' in status_text:
                raise Exception("BLAST query expired or does not exist.")
            elif 'Status=READY' in status_text:
                if 'ThereAreHits=yes' in status_text:
                    print("{request_data.question_uuid} results are ready, retrieving and saving...")
                    results_response = requests.get(base_url, params=get_results_params)
                    results_response.raise_for_status()
                    # Save the results to a file
                    with open(f'{save_path}/BLAST_results_{request_data.question_uuid}.txt', 'w') as file:
                        file.write(results_response.text)
                    print(f'Results saved in BLAST_results_{request_data.question_uuid}.txt')
                    break
                else:
                    print("No hits found.")
                    break
            else:
                raise Exception("Unknown status.")
        if attempt == max_attempts - 1:
            raise TimeoutError("Maximum attempts reached. Results may not be ready.")

def blast_result_file_extractor(question, file_path):
    """Extracting the answer from blast result file"""
    print('In result file extractor')
    #extract answer
    answer_extractor = AnswerExtractor()
    return answer_extractor.query(question, file_path)
