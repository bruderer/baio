# E-utilities Tool

NCBI E-utilities are a set of server-side programs that provide a stable interface to the Entrez query and database system at the National Center for Biotechnology Information (NCBI). E-utilities use a fixed URL syntax that translates a standard set of input parameters into the values necessary for various NCBI software components to search for and retrieve the requested data.

### Basic E-utilities Workflow:

1. **ESearch**: Searches and retrieves primary IDs (for use in EFetch, ELink, and ESummary) and term translations.
2. **EFetch**: Retrieves records in the requested format from a list of one or more primary IDs.
3. **ESummary**: Retrieves document summaries from a list of primary IDs.
4. **ELink**: Retrieves IDs and linknames from a list of one or more primary IDs.

The E-utilities tool in baio provides a interface to access various NCBI databases using the Entrez Programming Utilities (E-utilities).

### Currently Supported Databases Accessible via NCBI E-utilities:
- [Gene](https://www.ncbi.nlm.nih.gov/gene/): for gene-related queries
- [SNP](https://www.ncbi.nlm.nih.gov/snp/): for single nucleotide polymorphism information
- [OMIM](https://www.ncbi.nlm.nih.gov/omim): for data on genetic disorders and related genes

### Input
- Natural language queries related to genes, SNPs and genetic disorders.
- Specific identifiers (e.g., gene symbols, SNP rs numbers)

### Process
1. Query analysis and parsing
2. API call construction
3. Execution of API calls to NCBI E-utilities
4. Retrieval of results
5. Processing and formatting of retrieved data

### Output
- Structured data returned by the api (e.g., gene information, SNP details, OMIM entries).
- question, answer and path to returned data is logged in the JSON log file.
- Text summary: A concise, human-readable answer to the user's query, generated from the retrieved data.

### Questions that can be answered
- What is the official symbol for a given gene?
- What are the details of a specific SNP?
- What genetic disorders are associated with a particular gene?
- What is the genomic location of a gene?
- What are the protein products of a gene?

## Eutils Tool Logic

The E-utilities tool is composed of several Python files, each responsible for different aspects of the tool's functionality. Here's a breakdown of each file and its primary components:

### 1. api_form.py

This file defines the structure for API requests using Pydantic models.

#### Classes:
- `EutilsAPIRequest`: Defines the structure for general E-utilities API requests.
  - Fields include URL, database, search terms, and other API parameters.
- `EfetchRequest`: Specific structure for E-fetch API requests.
  - Includes fields for database, IDs, and return mode.

These classes are used as input for the langchain structured output function (see query_generator.py).

### 2. query_generator.py

This file is a critical component of the E-utilities tool, responsible for translating natural language queries into structured API requests. It leverages advanced natural language processing techniques and the Langchain library to achieve this translation.

#### Key Function:

`eutils_API_query_generator(question: str, llm, doc) -> EutilsAPIRequest`

This function is the core of the query generation process. It takes three main inputs:
1. `question`: The natural language query from the user.
2. `llm`: A language model (typically a Langchain ChatModel).
3. `doc`: A vector store containing relevant documentation or context.

#### Process Flow:

1. **Context Retrieval**:
   - Uses the vector store (`doc`) to retrieve relevant context for the query.
   - This context helps in understanding the domain-specific nuances of the query.

2. **Query Analysis**:
   - Utilizes Langchain's `create_structured_output_runnable` to create a pipeline for structured output generation.
   - This pipeline uses the provided language model (`llm`) to interpret the question and extract relevant parameters.

3. **Structured Query Generation**:
   - Constructs an `EutilsAPIRequest` object based on the interpreted query.
   - Fills in necessary fields like database selection, search terms, and other API parameters.

#### Example Usage:

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0)
embedding = OpenAIEmbeddings()
doc = FAISS.load_local("path_to_vector_store", embedding)

# Generate structured query
question = "What is the official gene symbol for LMP10?"
structured_query = eutils_API_query_generator(question, llm, doc)

# structured_query is now an EutilsAPIRequest object ready for use in API calls
```

This approach allows for flexible natural language understanding, translating user queries into structured API requests that can be easily processed by the NCBI E-utilities.

### 3. query_executer.py

This module handles the execution of API calls to NCBI E-utilities and processes the responses. It implements a two-step process for most queries: first using ESearch to get relevant IDs, then using EFetch (or ESummary for SNPs) to retrieve detailed information.

#### Key Functions:

- `execute_eutils_api_call(request_data: Union[EutilsAPIRequest, EfetchRequest])`:
  - Sends the API request to NCBI and handles the response.
  - Can handle both ESearch and EFetch requests.
  - Returns the response as JSON, raw bytes, or error message depending on the situation.

- `extract_id_list(response, retmode: str) -> List[int | str]`:
  - Extracts the list of IDs from an ESearch response.
  - Handles both JSON and XML response formats.

- `make_efetch_request(api_call: EutilsAPIRequest, id_list: List[int | str]) -> EfetchRequest`:
  - Creates an EFetchRequest object based on the initial ESearch results.

- `handle_non_snp_query(api_call: EutilsAPIRequest) -> List[Union[dict, str]]`:
  - Manages the flow for non-SNP queries:
    1. Executes ESearch to get relevant IDs.
    2. Creates and executes an EFetch request using those IDs.
  - Returns the EFetch response.

- `handle_snp_query(api_call: EutilsAPIRequest) -> List[Union[dict, str]]`:
  - Manages the flow for SNP queries:
    1. Executes ESearch to get relevant SNP IDs.
    2. Uses ESummary instead of EFetch for SNP data retrieval.
  - Returns the ESummary response.

- `save_response(response_list: List[Union[dict, str]], file_path: str, question_uuid: str) -> str`:
  - Saves the API response to a file for further processing or reference.
  - Handles different response types (JSON, binary, text).
  - Returns the filename of the saved response.

#### Flow of Operation:

1. The `execute_eutils_api_call` function is called with an EutilsAPIRequest for the initial ESearch.
2. The response is processed, and IDs are extracted using `extract_id_list`.
3. For non-SNP queries:
   - An EFetchRequest is created using `make_efetch_request`.
   - `execute_eutils_api_call` is called again with this EFetchRequest.
4. For SNP queries:
   - The ESummary endpoint is used instead of EFetch.
5. The final response (from EFetch or ESummary) is saved using `save_response`.

This two-step process allows for efficient querying of NCBI databases, first narrowing down the relevant entries and then fetching detailed information only for those entries.

### 4. answer_extractor.py

This module is responsible for processing the raw API responses from NCBI E-utilities and transforming them into concise, human-readable answers. It plays a crucial role in making the complex data returned by the API accessible and understandable to users.

#### Key Class:

`EutilsAnswerExtractor`

This class contains methods to parse and extract information from API responses, utilizing language models to generate summaries.

#### Key Method:

`query(self, question: str, file_path: str, llm, embedding) -> dict`

This method is the core of the answer extraction process. It takes four main inputs:
1. `question`: The original natural language query from the user.
2. `file_path`: Path to the file containing the API response.
3. `llm`: A language model (typically a Langchain ChatModel).
4. `embedding`: An embedding model for vector operations.

#### Process Flow:

1. **Data Loading**:
   - Reads the API response from the provided file path.
   - Handles different file types (JSON, XML, text) based on the API response format.

2. **Text Splitting**:
   - Uses Langchain's `CharacterTextSplitter` to break down large responses into manageable chunks.
   - This step is crucial for processing lengthy API responses within the context window of the language model.

3. **Embedding and Retrieval**:
   - Creates embeddings of the text chunks using the provided embedding model.
   - Stores these embeddings in a FAISS vector store for efficient retrieval.

4. **Question Answering**:
   - Utilizes Langchain's `ConversationalRetrievalChain` to generate an answer based on the question and the relevant parts of the API response.
   - This chain combines retrieval of relevant information with language model generation to produce coherent and accurate answers.

5. **Answer Formatting**:
   - Processes the generated answer to ensure it's in a user-friendly format.
   - Handles special cases for different types of queries (e.g., gene information, SNP details, OMIM entries).

#### Key Features:

1. **Customized Prompts**: Uses specially crafted prompts to guide the language model in extracting and summarizing relevant information from API responses.

2. **Context-Aware Processing**: Takes into account the original question to ensure the generated answer is relevant and on-topic.

3. **Flexible Parsing**: Capable of handling various response formats from different NCBI databases.

4. **Memory Management**: Utilizes Langchain's memory features to maintain context across multiple queries if needed.

#### Langchain Integration:

- **Text Splitting**: Uses `CharacterTextSplitter` for efficient processing of large text blocks.
- **Vector Storage**: Employs FAISS vector store through Langchain for quick and relevant information retrieval.
- **Retrieval Chain**: Leverages `ConversationalRetrievalChain` for context-aware question answering.
- **Prompt Engineering**: Utilizes Langchain's prompt templates for consistent and effective interaction with the language model.

#### Example Usage:

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0)
embedding = OpenAIEmbeddings()

# Create extractor instance
extractor = EutilsAnswerExtractor()

# Extract answer
question = "What is the function of the BRCA1 gene?"
file_path = "path/to/api_response.json"
result = extractor.query(question, file_path, llm, embedding)

print(result['answer'])
# Output: BRCA1 (Breast Cancer 1) is a tumor suppressor gene that helps repair DNA damage and maintain genomic stability...
```

This module ensures that the wealth of information returned by NCBI E-utilities is distilled into clear, concise, and relevant answers to user queries, making complex biological data more accessible.

### 5. tool.py

This file serves as the main entry point for the E-utilities tool, orchestrating the entire process from user input to final output. It brings together all the components of the tool to provide a seamless interface for querying NCBI databases.

#### Key Function:

`eutils_tool(question: str, llm, embedding) -> str`

This function is the primary interface for the E-utilities tool. It takes three main inputs:
1. `question`: The natural language query from the user.
2. `llm`: A language model (typically a Langchain ChatModel).
3. `embedding`: An embedding model for vector operations.

#### Process Flow:

1. **Initialization**:
   - Sets up necessary file paths and directories for logging and storing results.
   - Loads the vector store containing relevant documentation.

2. **Query Generation**:
   - Calls `eutils_API_query_generator` to convert the natural language question into a structured `EutilsAPIRequest` object.

3. **API Interaction**:
   - Determines whether to use `handle_non_snp_query` or `handle_snp_query` based on the query type.
   - Executes the appropriate function to interact with the NCBI E-utilities API.

4. **Response Processing**:
   - Saves the API response using `save_response` function.
   - Logs the query details and file paths for future reference.

5. **Answer Extraction**:
   - Uses `result_file_extractor` to process the saved API response and generate a human-readable answer.

6. **Result Return**:
   - Returns the final answer as a string.

#### Key Features:

1. **Error Handling**: Implements try-except blocks to gracefully handle potential errors during the process.

2. **Logging**: Maintains a detailed log of each query, including the question, generated file paths, and API URLs used.

3. **Flexible Database Handling**: Automatically adapts to different NCBI databases (Gene, SNP, OMIM) based on the query content.

4. **Caching**: Saves API responses, allowing for quick retrieval of previously queried information without repeating API calls.

#### Example Usage:

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0)
embedding = OpenAIEmbeddings()

# Use the tool
question = "What is the official gene symbol for LMP10?"
result = eutils_tool(question, llm, embedding)

print(result)
# Output: The official gene symbol for LMP10 is PSMB10 (Proteasome 20S Subunit Beta 10).
```

#### Integration with Other Modules:

`tool.py` integrates all other modules of the E-utilities tool:
- Uses `query_generator.py` for translating natural language to structured queries.
- Employs `query_executer.py` for making API calls and handling responses.
- Utilizes `answer_extractor.py` for processing API responses into human-readable answers.

This modular approach allows for easy maintenance and potential expansion to include more NCBI databases or query types in the future.