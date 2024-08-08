## 1. The API Form Class

### Key Components:

- **Pydantic BaseModel**: Inherits from `pydantic.BaseModel` for data validation and settings management.
- **Fields**: Each field corresponds to a parameter in the API call.
- **Field Types**: Specify the expected data type for each field (e.g., `str`, `int`, `Optional[str]`).
- **Default Values**: Provide default values for fields when applicable.
- **Field Descriptions**: Include detailed descriptions to guide the LLM in correctly populating the fields.


### Example:

```python
from pydantic import BaseModel, Field
from typing import Optional

class EutilsAPIRequest(BaseModel):
    url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        description="URL endpoint for the NCBI Eutils API"
    )
    db: str = Field(
        ...,
        description="Database to search. E.g., 'gene' for gene database, 'snp' for SNPs"
    )
    term: Optional[str] = Field(
        None,
        description="Search term. Used to query the database."
    )
    retmax: int = Field(..., description="Maximum number of records to return.")
    # ... other fields ...
```
   
## 2. The Query Generator
The query generator is responsible for creating a structured API request based on the user's question and relevant documentation.
Key Components:

- Langchain's create_structured_output_runnable: Creates a runnable that outputs a structured object based on the API form class.
- ChatPromptTemplate: Defines the prompt structure for the LLM.
- Relevant Documentation Retrieval: Uses embeddings and vector stores to fetch relevant API documentation.

### Example 

```python 
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate

def eutils_API_query_generator(question: str):
    structured_output_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class algorithm for extracting information in structured formats."),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format")
    ])
    
    runnable = create_structured_output_runnable(EutilsAPIRequest, llm, structured_output_prompt)
    
    # Retrieve relevant documentation
    retrieved_docs = eutils_doc.as_retriever().get_relevant_documents(question)
    top_3_retrieved_docs = "".join(doc.page_content for doc in retrieved_docs[:3])
    
    eutils_call_obj = runnable.invoke({
        "input": f"User question = {question}\nexample documentation: {top_3_retrieved_docs}"
    })
    
    return eutils_call_obj

```

### 3. API Call Execution
Once the structured API request is generated, it can be executed to make the actual API call.
Key Components:

- Request Preparation: Convert the structured object into URL parameters or JSON payload.
- HTTP Request: Use libraries like requests or urllib to make the API call.
- Response Handling: Process the API response and extract relevant information.

### Example:
```python
import urllib.request
import json

def make_api_call(request_data: EutilsAPIRequest):
    # Prepare query string
    query_params = request_data.dict(exclude={"url", "method", "headers"})
    encoded_query_string = urllib.parse.urlencode(query_params)
    
    # Construct full URL
    full_url = f"{request_data.url}?{encoded_query_string}"
    
    # Make the API call
    with urllib.request.urlopen(full_url) as response:
        response_data = response.read()
    
    # Process and return the response
    return json.loads(response_data)
```

