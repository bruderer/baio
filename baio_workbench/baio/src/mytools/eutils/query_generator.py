import uuid

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate
from pydantic import ValidationError

from . import EutilsAPIRequest


def eutils_API_query_generator(question: str, llm, doc):
    """FUNCTION to write api call for any BLAST query,"""

    BLAST_structured_output_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in "
                "structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input: "
                "{input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    runnable = create_structured_output_runnable(
        EutilsAPIRequest, llm, BLAST_structured_output_prompt
    )
    # retrieve relevant info to question
    retrieved_docs = doc.as_retriever().get_relevant_documents(question)
    # keep top 3 hits
    top_3_retrieved_docs = "".join(doc.page_content for doc in retrieved_docs[:3])
    eutils_call_obj = runnable.invoke(
        {
            "input": f"User question = {question}\nexample documentation: "
            f"{top_3_retrieved_docs}"
        }
    )
    eutils_call_obj.question_uuid = str(uuid.uuid4())
    try:
        query_request = eutils_call_obj
        query_request.url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        if query_request.db == "gene":
            query_request.term = format_search_term(query_request.term)
            query_request.retmode = "json"
        if query_request.db == "snp":
            query_request.url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            )
        print(f"Queryy is: {query_request}")
        # we set the url here, pipeline requires it to be esearch
        # Now you can use query_request as an instance of BlastQueryRequest
    except ValidationError as e:
        print(f"Validation error: {e}")
        # Handle validation error
    return query_request


def format_search_term(term, taxonomy_id=9606):
    """To be replaced with chain that fetches the taxonomy id of the requested "
    "organism!"""
    if term is None:
        return f"+AND+txid{taxonomy_id}[Organism]"
    else:
        return term + f"+AND+txid{taxonomy_id}[Organism]"
