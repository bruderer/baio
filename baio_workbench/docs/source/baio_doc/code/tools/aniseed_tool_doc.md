# Aniseed Tool

Aniseed (Ascidian Network for In Situ Expression and Embryological Data) is a database and web interface for ascidian developmental biology. It provides gene expression patterns, anatomical descriptions, and other developmental biology data for ascidian species.

## BaIO Aniseed Tool

The Aniseed tool in BaIO provides an advanced interface to query the Aniseed database using natural language. Unlike other tools in BaIO, the Aniseed tool implements a multi-step decision process to handle complex queries efficiently.

### Features:
- Natural language query processing
- Multi-step decision making for query handling
- Automatic selection and execution of multiple Aniseed API endpoints when necessary
- Query construction and submission to Aniseed
- Result retrieval, processing, and integration from multiple API calls
- Human-readable answer generation

### Input
- Natural language queries about ascidian gene expression, developmental stages, anatomical structures, or any combination thereof.

### Process
1. Query analysis and multi-step decision making
2. Aniseed API endpoint selection (potentially multiple)
3. API call construction and submission (potentially multiple calls)
4. Result retrieval and processing from all API calls
5. Integration of results from multiple calls (if applicable)
6. Answer generation from integrated Aniseed results

### Output
- Structured data returned by the Aniseed API (e.g., gene expression patterns, developmental stage information)
- Text summary: A concise, human-readable answer to the user's query, potentially integrating information from multiple API calls
- CSV file containing detailed results (when applicable)
- Logging of query, results, and generated answer

### Example Queries:
- "What genes are expressed in the notochord of Ciona intestinalis at the larval stage?"
- "Compare the expression patterns of Brachyury between the gastrula and neurula stages in Ciona intestinalis."
- "Find genes expressed in both the tail and the trunk of Ciona intestinalis embryos between stages 15 and 20."

## Aniseed Tool Implementation:

The Aniseed tool has a unique architecture within BaIO, reflecting its multi-step decision process:

### 1. aniseed/api_form.py
Defines multiple structures for different Aniseed API requests:
- `ANISEEDQueryRequest`: Base class for all Aniseed queries
- Specific request classes for each Aniseed API endpoint (e.g., `AllGenesQuery`, `AllGenesByStageQuery`, etc.)

### 2. aniseed/multi_step_decision.py
Implements the multi-step decision process:
- `AniseedStepDecider`: Determines which API calls are necessary to answer a query
- `ANISEED_multistep`: Orchestrates the multi-step process

### 3. aniseed/query_generator.py
Generates structured Aniseed queries based on the decisions made:
- `ANISEED_query_generator`: Creates specific API requests for each required endpoint

### 4. aniseed/query_executer.py
Handles the execution of Aniseed API calls:
- `execute_query`: Sends requests to the Aniseed API
- `save_ANISEED_result`: Saves results from each API call

### 5. aniseed/answer_extractor.py
Processes and integrates results from multiple API calls:
- `AniseedJSONExtractor`: Extracts relevant information from JSON responses
- `ANISEED_answer`: Generates the final answer by integrating all results

### 6. aniseed/tool.py
The main entry point for the Aniseed tool:
- `aniseed_tool`: Orchestrates the entire process, from query input to answer generation

## Unique Aspects of the Aniseed Tool:

1. **Multi-step Decision Making**: Unlike other tools, Aniseed may require multiple API calls to answer a single query. The tool decides which calls are necessary based on the query complexity.

2. **Flexible API Endpoint Usage**: The tool can dynamically select and use multiple Aniseed API endpoints as needed for a single query.

3. **Result Integration**: When multiple API calls are made, the tool integrates the results to provide a comprehensive answer.

4. **CSV Output**: For queries that return substantial data, the tool generates a CSV file for easier data analysis by the user.

## Usage of the Aniseed Tool

To use the Aniseed tool within BaIO:

```python
from langchain.chat_models import ChatOpenAI
from baio.src.mytools.aniseed import aniseed_tool

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Use the Aniseed tool
question = "What genes are expressed in the notochord of Ciona intestinalis between stages 15 and 20?"
result = aniseed_tool(question, llm)

print(result)
# Output: Based on the Aniseed database, the genes expressed in the notochord of Ciona intestinalis between stages 15 and 20 include Brachyury, Noto, and FoxA. A CSV file 'aniseed_results.csv' has been generated with detailed expression data for these genes.

# The tool also generates a CSV file with detailed results when applicable
```

The Aniseed tool's unique architecture allows it to handle complex queries about ascidian development, potentially integrating data from multiple Aniseed API endpoints to provide comprehensive answers. Its flexibility and multi-step approach make it a powerful tool for developmental biology researchers working with ascidian models.