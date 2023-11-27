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
import tempfile
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
from pydantic import BaseModel, Field, validator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json

# Lock for synchronizing file access
file_lock = threading.Lock()

os.environ["OPENAI_API_KEY"] = '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'
embeddings = OpenAIEmbeddings()

model_name = 'gpt-4'
llm = ChatOpenAI(model_name=model_name, temperature=0)


# ncbi_jin_db = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/faissdb", embeddings)
from langchain.document_loaders.csv_loader import CSVLoader
tmp_file_path = "/usr/src/app/baio/data/persistant_files/user_manuals/api_documentation/ucsc/ucsc_genomes_converted.csv"

loader = CSVLoader(
    file_path=tmp_file_path, 
    encoding="utf-8",
    csv_args={
        "delimiter": ",",
    }
)

data = loader.load()
embeddings = OpenAIEmbeddings()
ucsc_vectorstore = FAISS.from_documents(data, embeddings)
ucsc_retriever = ucsc_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

class BlastQueryRequest(BaseModel):
    url: str = Field(
        default="https://blast.ncbi.nlm.nih.gov/Blast.cgi?",
        description="URL endpoint for API calls, the NCBI BLAST API for general biology questions is default and  DNA alignnment to a specific genome use https://genome.ucsc.edu/cgi-bin/hgBlat?"
    )
    cmd: str = Field(
        default="Put",
        description="Command to execute, 'Put' for submitting query, 'Get' for retrieving results."
    )
    program: Optional[str] = Field(
        default="blastn",
        description="BLAST program to use, e.g., 'blastn' for nucleotide BLAST."
    )
    database: str = Field(
        default="nt",
        description="Database to search, e.g., 'nt' for nucleotide database."
    )
    query: Optional[str] = Field(
        None,
        description="Nucleotide or protein sequence for the BLAST or blat query, make sure to always keep the entire sequence given."
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
    ucsc_db: str = Field(
        default="hg38",
        description="Genome assembly to use in the UCSC Genome Browser, use the correct db for the organisms. Human:hsg38; Mouse:mm10; Dog:canFam6"
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
        Based on the explanation and example questions below:\n
        {context}
        Note: for DNA sequence alignment of an orgnaisms genome ALWAYS use: https://genome.ucsc.edu/cgi-bin/hgBlat?
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
    def query(self,  question: str, file_path: str) -> str:
        #we make a short extract of the top hits of the files
        first_400_lines = []
        with open(file_path, 'r') as file:
            for _ in range(400):
                line = file.readline()
                if not line:
                    break
                first_400_lines.append(line)
        # Create a temporary file and write the lines to it
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.writelines(first_400_lines)
            temp_file_path = temp_file.name       
        if os.path.exists(temp_file_path):
            print('found file')
            print(temp_file_path)
            loader = TextLoader(temp_file_path)
        else:
            print(f"Temporary file not found: {temp_file_path}")   
        # loader = TextLoader(temp_file_path)
        documents = loader.load()
        os.remove(temp_file_path)
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

import re

def extract_content_between_backticks(text):
    # Regular expression pattern for content within triple backticks
    pattern = r"```(.*?)```"

    # Search for matches
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the first match or None if no match is found
    return matches[0] if matches else None

def api_query_generator(question: str):
    """ function executing:
    1: text retrieval from ncbi_doc
    2: structured data from (1) to generate BlastQueryRequest object
    """
    ncbi_api = NCBIAPI()
    retrieved_docs = ucsc_retriever.get_relevant_documents(question+'if the question is not about a specific organism dont retrieve anything')
    ucsc_info = retrieved_docs[0].page_content
    relevant_api_call_info = ncbi_api.query(f'answer this question:{question}\nYou might need this information for a sequence alignemnt to a genome if the organisms name in the question matches this:{ucsc_info}. Here is more information you need to answer the question:')
# Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=BlastQueryRequest)
    # Prompt
    prompt = PromptTemplate(
        template="Answer the user query \n{format_instructions}\n{query}\n",
        input_variables=["query", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    # Run Retrieval and write strucutred output 
    _input = prompt.format_prompt(query=question)
    print('input to string:\n\n\n')
    print(_input.to_string())
    print('-----')
    model = OpenAI(temperature=0)
    #output is a json, annoying but we have to deal with it
    output = model(_input.to_string())
    # output = extract_content_between_backticks(_input.to_string())
    # output = _input.to_string()
    print('output processed:\n')
    print(output)
    print('\nDONE WITH OUTPUT ')
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
            'db': request_data.ucsc_db,
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

def log_question_uuid_json(question_uuid, question, file_name, file_path, log_file_path, full_url):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(log_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Initialize or load existing data
    data = []

    # Try reading existing data, handle empty or invalid JSON
    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
        try:
            with file_lock:
                with open(log_file_path, 'r') as file:
                    data = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON in {log_file_path}. Starting a new log.")

    # Construct the full file path
    full_file_path = os.path.join(file_path, file_name)

    # Add new entry with UUID, question, file name, and file path
    data.append({
        "uuid": question_uuid, 
        "question": question, 
        "file_name": file_name, 
        "file_path": full_file_path,
        "API_info": full_url
    })

    # Write updated data
    with file_lock:
        with open(log_file_path, 'w') as file:
            json.dump(data, file, indent=4)
 
def custom_json_serializer(data, indent=1):
    """
    Custom JSON serializer to put each key on a new line. Reduce token size of UCSC returns.
    """
    def serialize(obj, indent_level=0):
        spaces = ' ' * indent_level * indent
        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = [f'\n{spaces}"{key}": {serialize(value, indent_level + 1)}' for key, value in obj.items()]
            return '{' + ','.join(items) + f'\n{spaces[:-indent]}' + '}'
        elif isinstance(obj, list):
            if not obj:
                return '[]'
            items = [serialize(value, indent_level + 1) for value in obj]
            return '[\n' + f',\n'.join(f'{spaces}{item}' for item in items) + f'\n{spaces[:-indent]}]'
        else:
            return json.dumps(obj)

    return serialize(data)
         
def fetch_and_save_blast_results(request_data: BlastQueryRequest, blast_query_return: str, save_path: str , 
                                 question: str, log_file_path: str, wait_time: int = 15, max_attempts: int = 10000):
    request_data.question_uuid=str(uuid.uuid4())
    if 'ucsc' in request_data.url:
        file_name = f'BLAT_results_{request_data.question_uuid}.json'
        log_question_uuid_json(request_data.question_uuid,question, file_name, save_path, log_file_path,blast_query_return)
        response = requests.post(blast_query_return)
        response.raise_for_status()
        if response.status_code == 200:
            result_path = os.path.join(save_path, file_name)
            with open(result_path, 'w') as file:
                try:
                    blat_result = response.json()  
                    formatted_json = custom_json_serializer(blat_result)     
                    json.dump(formatted_json, file, indent=0)
                except json.JSONDecodeError:
                    # If it's not JSON, save the raw text
                    file.write(response.text)
                return blat_result 
        else:
            blat_result = f"Error: {response.status_code}"
            return blat_result
    if 'blast' in request_data.url:
        file_name = f'BLAST_results_{request_data.question_uuid}.txt'
        log_question_uuid_json(request_data.question_uuid,question, file_name, save_path, log_file_path,blast_query_return)        
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
                with open(f'{save_path}{file_name}', 'w') as file:
                    file.write("BLAST query FAILED.")
            elif 'Status=UNKNOWN' in status_text:
                with open(f'{save_path}{file_name}', 'w') as file:
                    file.write("BLAST query expired or does not exist.")
                raise 
            elif 'Status=READY' in status_text:
                if 'ThereAreHits=yes' in status_text:
                    print("{request_data.question_uuid} results are ready, retrieving and saving...")
                    results_response = requests.get(base_url, params=get_results_params)
                    results_response.raise_for_status()
                    # Save the results to a file
                    print(f'{save_path}{file_name}')
                    with open(f'{save_path}{file_name}', 'w') as file:
                        file.write(results_response.text)
                    print(f'Results saved in BLAST_results_{request_data.question_uuid}.txt')
                    break
                else:
                    # Writing "No hits found" to the file
                    with open(f'{save_path}{file_name}', 'w') as file:
                        file.write("No hits found")
                    break
            else:
                print('Unknown status')
                with open(f'{save_path}{file_name}', 'w') as file:
                    file.write("Unknown status")
                break 
        if attempt == max_attempts - 1:
            raise TimeoutError("Maximum attempts reached. Results may not be ready.")

def blast_result_file_extractor(question, file_path):
    """Extracting the answer from blast result file"""
    print('In blast result file extractor')
    #extract answer
    answer_extractor = AnswerExtractor()
    return answer_extractor.query(question, file_path)

##
###
####
#####
model_name = 'gpt-3.5-turbo-1106'
llm = ChatOpenAI(model_name=model_name, temperature=0)


log_file_path='/usr/src/app/baio/data/persistant_files/evaluation/251123_test_log_2/logfile.json'
save_file_path='/usr/src/app/baio/data/persistant_files/evaluation/251123_test_log_2/'
question= "Which organism does the DNA sequence come from:ATGTGCTTGTAGGAAGCAGCACAGGCCAGAAGAGGTTGTCAGATTCCCTAGAACTGGAGTTAGAAGCAGTTGTGAGCTCCTCTATGTAGGTGCTGAGAACTAAACCTGGATCCCATGAGCCATCTCCCTAA"

cost_questions = 0
with get_openai_callback() as cb:
    for question in question_list:
        query_request = api_query_generator(question)
        print('generated query\n\n\n')
        rid = submit_blast_query(query_request)
        print(f"Total cost is: {cb.total_cost} USD")

        error_list = []
        fetch_and_save_blast_results(query_request,rid, save_file_path, question, log_file_path)

        print(f"Total cost is: {cb.total_cost} USD")

        with file_lock:
            with open(log_file_path, 'r') as file:
                data = json.load(file)
            current_uuid = data[-1]['uuid']
        
        # Access the last entry in the JSON array
        last_entry = data[-1]
        # Extract the file path
        current_file_path = last_entry['file_path']
        result = blast_result_file_extractor(question, current_file_path)
        print(result['answer'])
        for entry in data:
            if entry['uuid'] == current_uuid:
                entry['answer'] = result['answer']
                break
        # Write the updated data back to the log file
        with file_lock:
            with open(log_file_path, 'w') as file:
                json.dump(data, file, indent=4)        
        print(f"Query cost is: {cb.total_cost} USD")
        cost_questions+=cb.total_cost
        print(f"Total cost is: {cost_questions} USD")
