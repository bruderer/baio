# ANISEED Database Query Workflow

## 1. User Input
- The process begins when a user submits a question about organisms in the ANISEED database.

## 2. Query Analysis
- The `ANISEED_multistep` function is called with the user's question.
  - It uses a language model to determine which API functions are needed to answer the question.
  - It also retrieves relevant context from a vector database (FAISS).

## 3. API Query Generation
- For each required API function:
  - The `ANISEED_query_generator` is called.
  - It constructs an `ANISEEDQueryRequest` object with the necessary parameters.
  - This object is used to generate a full URL for the API call.

## 4. API Calls
- The program makes HTTP GET requests to the ANISEED API using the generated URLs.
- The JSON responses are saved to temporary files.

## 5. Data Processing
- For each API response:
  - The JSON data is read from the temporary file.
  - An `AniseedJSONExtractor` object is created.
  - It generates a prompt for converting the JSON to a flat CSV format.
  - A language model is used to generate Python code based on this prompt.
  - The generated code is executed to convert the JSON to a CSV file.

## 6. Error Handling and Retries
- If the JSON to CSV conversion fails:
  - The process is retried up to 5 times.
  - Each retry includes the previous error message to help the language model correct the issue.

## 7. Output
- The paths to the generated CSV files are returned.

## Key Components:

1. **AniseedAPI**: Class containing methods for different ANISEED API endpoints.
2. **AniseedStepDecider**: Determines which API functions to use based on the question.
3. **ANISEED_multistep**: Decides on API calls and retrieves relevant context.
4. **ANISEED_query_generator**: Constructs API queries.
5. **AniseedJSONExtractor**: Handles JSON to CSV conversion.
6. **LLM (Language Model)**: Used for decision-making and code generation.
7. **FAISS Vector Store**: Stores and retrieves relevant context for queries.

## Data Flow:

## In depth problem 
- Aniseed has a limited query structure to filter data via API calls
- Aniseeds API calls require different fields and structures dependant on the method 

Because user questions are often time more granular than API calls, they can require information from multiple API endpoints. Hence a strategic decision of the sequence of API calls must be made. 
Once the different API calls executed the retrieved information can be filtered to extract the relevant information.

## 1: Decide what methods must be executed

### Method

Structured output step decider: 

- Fields = aniseed api call methods 

Output: Up to 3 api methods that have to be run 

## 2 Fill the API form with appropriate query terms 

Input: Up to 3 API methods that have to be executed; Aniseed doc

Process: Each API method has an **API FORM** class
