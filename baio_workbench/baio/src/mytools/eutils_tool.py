from typing import Optional, Dict, List
from pydantic import BaseModel, Field
import requests
import urllib.request
import urllib.parse
import json
import time
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
from typing import Optional, Any, Dict
import requests
import time
import re
import requests
from pydantic import ValidationError
import time
from pydantic import BaseModel, Field, validator
from typing import Optional, Union
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import faiss
loader = TextLoader("/usr/src/app/baio/data/persistant_files/user_manuals/api_documentation/ncbi/jin_et_al.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

ncbi_jin_db = FAISS.from_documents(docs, embeddings)

os.environ["OPENAI_API_KEY"] = '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'
embeddings = OpenAIEmbeddings()

model_name = 'gpt-4'
llm = ChatOpenAI(model_name=model_name, temperature=0)

###class to be filled by chain:
class EutilsAPIRequest(BaseModel):
    url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        description="URL endpoint for the NCBI Eutils API."
    )
    method: str = Field(
        default="GET",
        description="HTTP method for the request. Typically 'GET' or 'POST'."
    )
    headers: Dict[str, str] = Field(
        default={"Content-Type": "application/json"},
        description="HTTP headers for the request. Default is JSON content type."
    )
    db: str = Field(
        ...,
        description="Database to search. E.g., 'gene' for gene database, 'snp' for SNPs, 'omim' for genetic diseases "
    )
    retmax: int = Field(
        ...,
        description="Maximum number of records to return."
    )
    retmode: str = Field(
        default="json",
        description="Return mode, determines the format of the response. Commonly 'json' or 'xml'."
    )
    sort: Optional[str] = Field(
        default="relevance",
        description="Sorting parameter. Defines how results are sorted."
    )
    term: Optional[str] = Field(
        None,
        description="Search term. Used to query the database. if it is for a SNP always remove rs before the number"
    )
    id: Optional[str] = Field(
        None,
        description="Database identifier(s) for specific records."
    )
    body: Optional[str] = None
    response_format: str = Field(
        default="json",
        description="Expected format of the response, such as 'json' or 'xml'."
    )
    # parse_keys: List[str] = Field(
    #     ...,
    #     description="Keys to parse from the response. Helps in extracting specific data."
    # )

class NCBIAPI2:
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
class NCBIAPI:
    """another test, if fails use ncbiapi2 """
    # def __init__(self):
    #     self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    #     template_api_ncbi = """
    #     You have to provide the necessary information to answer this question 
    #     Question: {question}\n\n
    #     Based on the explanaitions and example questions below:\n
    #     {context}
    #     """
    #     self.ncbi_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template_api_ncbi)
    def query(self, question: str) -> str:
        retriever = ncbi_jin_db.as_retriever()
        retrieved_docs = retriever.invoke(question)
        relevant_api_call_info = retrieved_docs[0].page_content
        return relevant_api_call_info


class EfetchRequest(BaseModel):
    url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        description="URL endpoint for the NCBI Efetch API."
    )
    db: str = Field(
        ...,
        description="Database to fetch from. E.g., 'gene' for gene database."
    )
    id: str = Field(
        ...,
        description="Comma-separated list of NCBI record identifiers."
    )
    retmode: str = Field(
        default="xml",
        description="Return mode, determines the format of the response. Commonly 'xml' or 'json'."
    )
    rettype: Optional[str] = Field(
        None,
        description="Return type, determines the type of data to return. Specific to the database."
    )
    
def extract_ids_from_response(response: Dict[str, Any]) -> str:
    """ Extracts IDs from the esearch response and returns a comma-separated string. """
    id_list = response.get('esearchresult', {}).get('idlist', [])
    return ','.join(id_list)

def api_query_generator(question: str):
    """ NEW VERSION FIRST IN THE PIPE :
    function executing:
    1: text retrieval from ncbi_doc
    2: structured data from (1) to generate EutilsAPIRequest object
    """
    ncbi_api = NCBIAPI()
    relevant_api_call_info = ncbi_api.query(question)
# Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=EutilsAPIRequest)
    # Prompt
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    print('the prompt is the following:\n\n')
    # Run Retrieval and write strucutred output 
    _input = prompt.format_prompt(query=f'The users question you have is the following, always use it as search terms: {question}\n\n\
                                  The following is context with example questions:{relevant_api_call_info}')
    print(_input.to_string())
    model = OpenAI(temperature=0)
    #output is a json, annoying but we have to deal with it
    output = model(_input.to_string())
    # print('the output is:\n')
    # print(output)
    output = parser.parse(output)
    try:
        query_request = output
        query_request.url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        print(query_request)
        #we set the url here, pipeline requires it to be esearch
        
        # Now you can use query_request as an instance of BlastQueryRequest
    except ValidationError as e:
        print(f"Validation error: {e}")
        return ['Failed to write API query instructions', output]
        # Handle validation error
    return query_request



#
def make_api_call(request_data: Union[EutilsAPIRequest, EfetchRequest]):
    """WOOOUHHH works """
    # Default values for optional fields
    default_headers = {"Content-Type": "application/json"}
    default_method = "GET"
    # Prepare the query string
    query_params = request_data.dict(exclude={"url", "method", "headers", "body", "response_format", "parse_keys"})
    if request_data.db == "omim":
        query_params = request_data.dict(exclude={"url", "method", "headers", "body", "response_format", "parse_keys", "retmod"})
        if request_data.id != '' or None:
            request_data.url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'

        # request_data.url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    query_string = urllib.parse.urlencode(query_params)
    # Construct the full URL
    full_url = f"{request_data.url}?{query_string}"
    print(f'Requesting: {full_url}')
    # Use provided headers or default if not available
    headers = getattr(request_data, 'headers', default_headers)
    # Use provided method or default if not available
    method = getattr(request_data, 'method', default_method)
    # Create and send the request
    req = urllib.request.Request(full_url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            response_data = response.read()
            #some db efetch do not return data as json, but we try first to extract the json
            try:
                if request_data.retmode.lower() == "json":
                    return json.loads(response_data)
            except:
                return response_data
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        raise
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        raise
 

#intake question
question = 'What are genes related to Meesmann corneal dystrophy'
question = 'What are genes related to Hemolytic anemia due to phosphofructokinase deficiency?'
question = 'What are genes related to Pseudohypoparathyroidism Ic?'

question = "Convert ENSG00000215251 to official gene symbol."
#write api call 1
call_1 = api_query_generator(question)
call_1


#make esearch api call 
response = make_api_call(call_1)
id_list = response.get('esearchresult', {}).get('idlist', [])
efetch_response_list = []
for id in id_list:
    #define efetch api call 
    efetch_request = EfetchRequest(db=call_1.db,
                                id=id,
                                retmode=call_1.retmode)
    #execute efetch api call to get final xml file 
    efetch_response = make_api_call(efetch_request)
    efetch_response_list.append(efetch_response)

efetch_response_list
