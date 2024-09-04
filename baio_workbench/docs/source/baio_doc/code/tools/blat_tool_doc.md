# BLAT Tool

BLAT (BLAST-Like Alignment Tool) is a pairwise sequence alignment algorithm designed to quickly find sequences of high similarity. It is particularly useful for aligning DNA sequences to a genome, or finding the genomic location of a given sequence.

### Basic BLAT Workflow:

1. **Query Submission**: A DNA or protein sequence is submitted as a query.
2. **Index Search**: The query is compared against an indexed database of the genome.
3. **Alignment**: Detailed alignments are performed for regions of high similarity.
4. **Result Generation**: A report is generated showing the genomic locations and details of matches.

### Baio BLAT implementation

The BLAT tool in BaIO provides an interface to access the UCSC BLAT service using natural language queries.

#### Features:
- Natural language query processing
- Automatic selection of appropriate genome assembly
- Query construction and submission to UCSC BLAT
- Result retrieval and parsing
- Human-readable answer generation

#### Input:
- Natural language queries describing the desired BLAT search and the DNA sequence.

#### Process:
1. Query analysis and parsing
2. Genome assembly selection
3. Query sequence extraction
4. BLAT API call construction and submission
5. Result retrieval and processing
6. Answer generation from BLAT results

#### Output:
- Structured data returned by the BLAT API (e.g., genomic locations, alignment details).
- Text summary: A concise, human-readable answer to the user's query, generated from the BLAT results.
- Logging of query, results, and generated answer.

#### Example Query:
"Align the DNA sequence ATCGATCGATCGATCG to the human genome."

### BLAT Tool Implementation:

The BLAT tool follows a similar structure to the BLAST tool:

1. `blat/api_form.py`: Defines the structure for BLAT API requests.
2. `blat/query_generator.py`: Generates structured BLAT queries from natural language input.
3. `blat/query_executer.py`: Handles the execution of BLAT API calls and processing of responses.
4. `blat/answer_extractor.py`: Processes the raw BLAT results and generates human-readable answers.
5. `blat/tool.py`: The main entry point for the BLAT tool.
