import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from baio.src.mytools.aniseed import (
    ANISEED_multistep,
    ANISEED_query_generator,
    AniseedStepDecider,
)

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)
embedding = OpenAIEmbeddings(api_key=openai_api_key)
ANISEED_db = FAISS.load_local(
    "./baio/data/persistant_files/vectorstores/aniseed", embedding
)


def test_ANISEED_multistep_real():
    # Test cases
    test_cases = [
        (
            "What genes are expressed in Ciona intestinalis between stage 1 and 10?",
            ["all_genes_by_stage_range"],
        ),
        # (
        #     "What are the regulatory regions in Ciona intestinalis?",
        #     ["all_regulatory_regions"],
        # ),
        (
            "What genes are expressed in the tail of Ciona intestinalis at the larval "
            "stage?",
            ["all_genes_by_stage", "all_genes_by_territory"],
        ),
        # ("What are all the genes in Ciona intestinalis?", ["all_genes"]),
        # (
        #     "What molecular tools are available for studying Ciona intestinalis?",
        #     ["all_molecular_tools"],
        # ),
    ]

    for question, expected_functions in test_cases:
        print(f"\nTesting question: {question}")
        result, top_3_docs = ANISEED_multistep(question, llm, ANISEED_db)

        # Assertions
        assert isinstance(
            result, AniseedStepDecider
        ), "Result should be an instance of AniseedStepDecider"
        assert isinstance(top_3_docs, str), "top_3_docs should be a string"
        print(f"One_or_more_steps: {result.One_or_more_steps}")
        print(f"functions_to_use_1: {result.functions_to_use_1}")
        print(f"functions_to_use_2: {result.functions_to_use_2}")
        print(f"functions_to_use_3: {result.functions_to_use_3}")

        # Check if all expected functions are in the function slots
        actual_functions = [
            result.functions_to_use_1,
            result.functions_to_use_2,
            result.functions_to_use_3,
        ]
        actual_functions = [func for func in actual_functions if func is not None]

        for expected_function in expected_functions:
            assert (
                expected_function in actual_functions
            ), f"Expected function {expected_function} not found in results"

        # now we write the api calls
        api_calls = []
        for function in actual_functions:
            print(f"\033[1;32;40mAniseed tool is treating: {function} \n")
            api_calls.append(
                ANISEED_query_generator(question, function, top_3_docs, llm)
            )


test_ANISEED_multistep_real()
