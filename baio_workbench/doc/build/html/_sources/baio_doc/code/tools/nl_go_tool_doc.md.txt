# Natural language to GO terms tool

This module provides functionality to process natural language queries about gene ontology (GO) terms. It defines a `NaturalLanguageExtractors` class for extracting gene names from natural language input and a `go_nl_query_tool` function that serves as the main entry point for GO annotation queries.

## Classes

### NaturalLanguageExtractors

This class contains methods to extract certain information in a structured manner from natural language input.

#### Methods:

##### __init__(self, natural_language_string)
- Initializes the extractor with the input string.
- Parameters:
  - `natural_language_string` (str): The string from which information will be extracted.

##### gene_name_extractor(self, llm) -> list
- Extracts gene names from the input string.
- Parameters:
  - `llm`: The language model to use for extraction.
- Returns:
  - A list of extracted gene names.

## Functions

### go_nl_query_tool(nl_input: str, llm, embedding) -> pd.DataFrame

This is the main function used when the input is a natural language written query containing gene names that need GO annotation.

#### Parameters:
- `nl_input` (str): A natural language string containing gene names to be processed.
- `llm`: The language model to use for processing.
- `embedding`: The embedding model to use.

#### Returns:
- `pd.DataFrame`: A DataFrame containing annotated genes with GO terms and gene IDs from mygene.

#### Process:
1. Creates an instance of `NaturalLanguageExtractors` with the input string.
2. Extracts gene names from the input using `gene_name_extractor`.
3. Creates a `GoFormater` instance with the extracted gene list.
4. Uses the `GoFormater` to annotate the genes with GO terms.
5. Parses RefSeq IDs in the resulting DataFrame.
6. Saves the result as a CSV file.
7. Returns the head of the final DataFrame.

## Usage

To use this module:

1. Import the necessary function:
   ```python
   from baio.src.mytools.nl_go_tool import go_nl_query_tool
   ```

2. Prepare your natural language input:
   ```python
   nl_input = "What are the GO terms for genes BRCA1, TP53, and EGFR?"
   ```

3. Call the `go_nl_query_tool` function:
   ```python
   result_df = go_nl_query_tool(nl_input, llm, embedding)
   ```

4. The result will be a DataFrame containing GO annotations for the specified genes:
   ```python
   print(result_df)
   ```

## Notes

- The module uses the `GoFormater` class from `baio.src.non_llm_tools` to perform the actual GO annotation.
- The results are saved as a CSV file in the `./baio/data/output/gene_ontology/` directory.
- The function returns only the head of the DataFrame for preview purposes. The full results are available in the saved CSV file.
- This tool is particularly useful for biologists and researchers who want to quickly obtain GO annotations for a set of genes using natural language queries.

This module is a key component of the BaIO system, allowing users to easily obtain gene ontology information using natural language queries, bridging the gap between human language and structured biological data.