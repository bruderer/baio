# BLAST Tool

BLAST (Basic Local Alignment Search Tool) is a sequence similarity search program that can be used to quickly search sequence databases for optimal local alignments to a query. The NCBI BLAST finds regions of similarity between biological sequences, comparing nucleotide or protein sequences to sequence databases and calculating the statistical significance.

## Basic BLAST Workflow:

1. **Query Submission**: A nucleotide or protein sequence is submitted as a query.
2. **Database Search**: The query is compared against a selected sequence database.
3. **Alignment**: Local alignments are performed to find regions of similarity.
4. **Scoring**: Alignments are scored and ranked based on statistical significance.
5. **Results**: A report is generated showing the best matches and their statistics.

## BaIO BLAST Tool

The BLAST tool in BaIO provides an interface to access the NCBI BLAST service using natural language queries.

### Supported BLAST Programs:
- blastn: Nucleotide-nucleotide BLAST
- blastp: Protein-protein BLAST
<!-- - blastx: Nucleotide query vs. protein database
- tblastn: Protein query vs. nucleotide database
- tblastx: Translated nucleotide vs. translated nucleotide -->

### Features:
- Natural language query processing
- Automatic BLAST program selection
- Query construction and submission to NCBI BLAST
- Result retrieval and parsing
- Human-readable answer generation

### Input
- Natural language queries describing desired search and the DNA or protein sequences.

### Process
1. Query analysis and parsing
2. BLAST program and database selection
3. Query sequence extraction (if provided in the natural language input)
4. BLAST API call construction and submission
5. Result retrieval and processing
6. Answer generation from BLAST results

### Output
- Structured data returned by the BLAST API (e.g., alignments, scores, E-values).
- Text summary: A concise, human-readable answer to the user's query, generated from the BLAST results.
- Logging of query, results, and generated answer.

### Example Queries:
- "What organism does this protein sequence likely come from: MAEGEITTFTALTEKFNLPPGNYKKPKLLY"
- "Are there any known genes similar to this sequence in mice?"

## BLAST Tool Implementation

The BLAST tool is implemented through several Python modules, each handling a specific part of the process:

### 1. blast/api_form.py

Defines the structure for BLAST API requests using Pydantic models.

#### Key Class:
`BlastQueryRequest`: Defines the structure for BLAST API requests.
- Fields include BLAST program, database, query sequence, and other BLAST parameters.

### 2. blast/query_generator.py

Responsible for generating structured BLAST queries from natural language input.

#### Key Function:
`BLAST_api_query_generator(question: str, doc, llm) -> BlastQueryRequest`

Process:
1. Analyzes the natural language question using the language model.
2. Extracts relevant parameters (sequence, desired database, etc.).
3. Constructs a `BlastQueryRequest` object.

### 3. blast/query_executer.py

Handles the execution of BLAST API calls and processing of responses.

#### Key Functions:
- `submit_blast_query(request_data: BlastQueryRequest) -> str`: 
  Submits the BLAST query and returns a Request ID (RID).
- `fetch_and_save_blast_results(request_data: BlastQueryRequest, rid: str, ...) -> str`:
  Retrieves BLAST results using the RID and saves them to a file.

### 4. blast/answer_extractor.py

Processes the raw BLAST results and generates human-readable answers.

#### Key Class:
`BLASTAnswerExtractor`: Contains methods to parse BLAST results and generate summaries.

#### Key Method:
`query(self, question: str, file_path: str, n: int, embedding, llm) -> dict`

Process:
1. Reads BLAST results from the file.
2. Uses language models to interpret results in the context of the original question.
3. Generates a concise, relevant answer.

### 5. blast/tool.py

The main entry point for the BLAST tool, orchestrating the entire process.

#### Key Function:
`blast_tool(question: str, llm, embedding) -> str`

Process Flow:
1. Generates a structured BLAST query from the natural language input.
2. Submits the BLAST query and retrieves results.
3. Processes the results to extract a relevant answer.
4. Returns the answer as a string.

## Usage

To use the BLAST tool within BaIO:

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from baio.src.mytools.blast import blast_tool

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0)
embedding = OpenAIEmbeddings()

# Use the tool
question = "What organism does this DNA sequence likely come from: ATCGATCGTAGCTAGC"
result = blast_tool(question, llm, embedding)

print(result)
# Output: Based on the BLAST results, the DNA sequence ATCGATCGTAGCTAGC likely comes from Homo sapiens (humans). The sequence shows...
```

This modular structure allows for easy maintenance and potential future expansions, such as supporting additional BLAST programs or customizing result interpretation for specific use cases.