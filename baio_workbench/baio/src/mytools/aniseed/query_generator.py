import uuid

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate

from . import AniseedAPI, ANISEEDQueryRequest, execute_query


def ANISEED_query_generator(
    question: str, function: str, top_3_retrieved_docs: str, llm
):
    """FUNCTION to write api call for any BLAST query,"""
    print("     Generating the query url...\n")

    structured_output_prompt = ChatPromptTemplate.from_messages(
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
        ANISEEDQueryRequest,
        llm,
        structured_output_prompt,
    )
    aniseed_call_obj = runnable.invoke(
        {
            "input": f"you have to answer this {question} by using this {function} and "
            f"fill in all fields based on {top_3_retrieved_docs}. NEVER use an argument"
            " that is not in the API documentation OR in the function input arguments"
        }
    )
    api = AniseedAPI()
    full_url = execute_query(api, aniseed_call_obj)
    print(f"The full url is: {full_url}")
    aniseed_call_obj.full_url = full_url
    aniseed_call_obj.question_uuid = str(uuid.uuid4())
    print(aniseed_call_obj)
    return aniseed_call_obj
