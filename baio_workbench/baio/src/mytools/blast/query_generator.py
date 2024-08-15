import uuid

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate

from . import BlastQueryRequest


def BLAST_api_query_generator(question: str, doc, llm):
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
        BlastQueryRequest, llm, BLAST_structured_output_prompt
    )
    # retrieve relevant info to question
    retrieved_docs = doc.as_retriever().get_relevant_documents(
        question
        + "if the question is not about a specific organism dont retrieve anything"
    )
    # keep top 3 hits
    top_3_retrieved_docs = "".join(doc.page_content for doc in retrieved_docs[:3])
    blast_call_obj = runnable.invoke(
        {"input": f"{question} based on {top_3_retrieved_docs}"}
    )
    blast_call_obj.question_uuid = str(uuid.uuid4())
    return blast_call_obj
