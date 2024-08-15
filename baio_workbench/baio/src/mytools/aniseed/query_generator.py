import uuid

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate

from . import AniseedAPI, get_query_class


def execute_query(api, query):
    # Get all fields from the query object
    all_params = query.dict()
    print(f"All parameters (for logging): {all_params}")

    # Create a copy of all params for the API call
    api_params = all_params.copy()

    # Remove fields that are not API parameters
    api_params.pop("full_url", None)
    api_params.pop("question_uuid", None)
    api_params.pop("required_function", None)

    # Remove any parameters that are None
    api_params = {k: v for k, v in api_params.items() if v is not None}

    print(f"Parameters for API call: {api_params}")

    # Get the required function from the API
    func = getattr(api, query.required_function)
    print(f"Function: {func}")

    # Call the function with the parameters
    return func(**api_params)


def ANISEED_query_generator(
    question: str,
    function: str,
    doc: str,
    llm,
):
    """FUNCTION to write api call for any BLAST query,"""
    print(f"     Generating the query url with function {function}...\n")
    QueryClass = get_query_class(function)

    structured_output_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in "
                "structured formats.",
            ),
            (
                "human",
                "ONLY USE THE FUNCTION NAME in  the 'required_function' field"
                "Use the given format to extract information from the following input: "
                "{input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    print(QueryClass)
    runnable = create_structured_output_runnable(
        QueryClass,
        llm,
        structured_output_prompt,
    )
    aniseed_call_obj = runnable.invoke(
        {
            "input": f"you have to answer this {question} by using this {function} and "
            f"fill in all fields based on {doc}. ONLY USE ARGUMENTS "
            "THAY ARE the function input arguments"
        }
    )
    api = AniseedAPI()
    full_url = execute_query(api, aniseed_call_obj)
    print(f"The full url is: {full_url}")
    aniseed_call_obj.full_url = full_url
    aniseed_call_obj.question_uuid = str(uuid.uuid4())
    print(aniseed_call_obj)
    return aniseed_call_obj
