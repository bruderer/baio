import uuid
from urllib.parse import urlencode

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate

from . import BLATdb, BLATQueryRequest


def BLAT_api_query_generator(
    question: str,
    llm,
    doc,
):
    """FUNCTION to write api call for any BLAT query,"""

    BLAT_structured_output_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in "
                "structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input, "
                "for human always use hg38 and only use specific regions if you "
                "get it prompted (e.g. only 'chromosome' is not sufficient): "
                "Always just align the dna sequence to the human genome if you are asked"
                "The DNA sequence ATA... is on the human genome chromosome:"
                "{input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    runnable_BLATQueryRequest = create_structured_output_runnable(
        BLATQueryRequest, llm, BLAT_structured_output_prompt
    )
    runnable_BLATdb = create_structured_output_runnable(
        BLATdb, llm, BLAT_structured_output_prompt
    )
    # retrieve relevant info to question
    retrieved_docs_data_base = doc.as_retriever().get_relevant_documents(question)
    retrieved_docs_query = doc.as_retriever().get_relevant_documents(question)
    # print(retrieved_docs_data_base[0])
    # keep top 3 hits
    top_3_retrieved_docs = "".join(doc.page_content for doc in retrieved_docs_query[:3])
    BLAT_db_obj = runnable_BLATdb.invoke(
        {
            "input": f"User question = {question}\nexample documentation: "
            f"{retrieved_docs_data_base}"
        }
    )
    print(f"Database used:{BLAT_db_obj.ucsc_db}")
    BLAT_call_obj = runnable_BLATQueryRequest.invoke(
        {
            "input": f"User question = {question}\nexample documentation: "
            f"{top_3_retrieved_docs}"
        }
    )
    BLAT_call_obj.ucsc_db = BLAT_db_obj.ucsc_db
    BLAT_call_obj.question_uuid = str(uuid.uuid4())
    data = {
        # "url": "https://genome.ucsc.edu/cgi-bin/hgBlat?",
        "userSeq": BLAT_call_obj.query,
        "type": BLAT_call_obj.ucsc_query_type,
        "db": BLAT_call_obj.ucsc_db,
        "output": "json",
    }
    # Build the API call
    query_string = urlencode(data)
    # Combine base URL with the query string
    full_url = f"{BLAT_call_obj.url}?{query_string}"
    BLAT_call_obj.full_url = full_url
    BLAT_call_obj.question_uuid = str(uuid.uuid4())
    return BLAT_call_obj
