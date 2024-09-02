# BaIO Agents Documentation

## General Overview of BaIO Agents

BaIO (Bridge to Biological Databases) uses a system of specialized agents to handle various types of biological data queries and tasks. Each agent is designed to interact with specific databases or perform particular functions.

### How Agents Work in General

1. **Input Processing**: Agents receive natural language queries from users.

2. **Task Identification**: The system determines which agent is best suited to handle the query using the `select_best_fitting_tool` function.

3. **Query Execution**: The selected agent processes the query, often by:
   - Formulating appropriate API calls to biological databases
   - Extracting relevant information from the responses
   - Performing data analysis or annotation

4. **Result Formatting**: Agents format the results into a user-friendly output, often as pandas DataFrames or structured text.

5. **Response**: The formatted results are returned to the user through the BaIO interface.

Now, let's look at each agent in detail:

## 1. ANISEED Agent

**File**: `baio/src/agents/aniseed_agent.py`

### Purpose
Handles queries related to the ANISEED (Ascidian Network for In Situ Expression and Embryological Data) database.

### Key Functions
- `aniseed_agent(question: str, llm)`

### Process
1. Receives a natural language question about ANISEED data.
2. Uses the ANISEED API to query the database.
3. Processes and formats the response.
4. Returns a list of file paths containing the query results.

### Usage Example
```python
result = aniseed_agent("What genes are expressed in Ciona intestinalis between stage 1 and 3?", llm)
```

## 2. BaIO Agent

**File**: `baio/src/agents/baio_agent.py`

### Purpose
The main agent that coordinates between different tools and databases, including NCBI services like BLAST and E-utilities and api calls to the UCSC genome browser.

### Key Functions
- `baio_agent(question: str, llm, embedding)`

### Process
1. Analyzes the question to determine the most appropriate tool.
2. Delegates the query to the selected tool (e.g., BLAST, E-utilities).
3. Processes the response and formats it for the user.

### Usage Example
```python
result = baio_agent("What is the protein sequence for the gene BRCA1?", llm, embedding)
```

## 3. CSV Chatter Agent

**File**: `baio/src/agents/csv_chatter_agent.py`

### Purpose
Allows users to interact with and analyze CSV files using natural language queries.

### Key Functions
- `csv_chatter_agent(question, file_paths: list, llm)`

### Process
1. Loads the specified CSV file(s).
2. Interprets the natural language query about the data.
3. Performs the requested analysis or data manipulation.
4. Returns the results, often as a pandas DataFrame.

### Usage Example
```python
result = csv_chatter_agent("What are the top 5 expressed genes in this dataset?", ["/path/to/data.csv"], llm)
```

## 4. File Annotator Agent

**File**: `baio/src/agents/file_annotator_agent.py`

### Purpose
Annotates genes in input files with additional information, such as GO terms.

### Key Functions
- `file_annotator_agent(llm, memory)`

### Process
1. Reads the input file containing gene names or IDs.
2. Fetches relevant annotations (e.g., GO terms) for each gene.
3. Adds the annotations to the original data.
4. Saves and returns the annotated file.

### Usage Example
```python
annotated_data = file_annotator_agent(llm, memory).run("Annotate genes in file1.csv with GO terms")
```

## 5. NL GO Agent

**File**: `baio/src/agents/nl_go_agent.py`

### Purpose
Processes natural language queries about Gene Ontology (GO) terms.

### Key Functions
- `go_nl_agent(question, llm)`

### Process
1. Extracts gene names from the natural language query.
2. Fetches GO terms for the identified genes.
3. Formats the GO information into a readable output.

### Usage Example
```python
go_info = go_nl_agent("What are the GO terms for BRCA1 and TP53?", llm)
```

These agents work together within the BaIO system to provide a comprehensive interface for biological data querying and analysis. They leverage natural language processing and specialized biological knowledge to bridge the gap between user queries and complex biological databases and tools.