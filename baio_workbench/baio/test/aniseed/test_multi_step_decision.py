import os

import pytest
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from baio.src.mytools.aniseed import ANISEED_multistep, AniseedStepDecider

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)

embedding = OpenAIEmbeddings(api_key=openai_api_key)
ANISEED_db = FAISS.load_local(
    "./baio/data/persistant_files/vectorstores/aniseed", embedding
)


def test_ANISEED_multistep_real():
    # Test cases
    test_cases = [
        (
            "What genes are expressed in the notochord of Ciona intestinalis?",
            "all_genes_by_territory",
        ),
        (
            "What are the regulatory regions for the MESP1 gene in Ciona "
            "intestinalis?",
            "all_regulatory_regions",
        ),
        (
            "What genes are expressed in the tail of Ciona intestinalis at the larval "
            "stage?",
            "all_genes_by_stage",
        ),
        ("What are all the genes in Ciona intestinalis?", "all_genes"),
        (
            "What molecular tools are available for studying Ciona intestinalis?",
            "all_molecular_tools",
        ),
    ]

    for question, expected_function in test_cases:
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

        # Check if the expected function is in one of the function slots
        assert any(
            func == expected_function
            for func in [
                result.functions_to_use_1,
                result.functions_to_use_2,
                result.functions_to_use_3,
            ]
            if func is not None
        ), f"Expected function {expected_function} not found in results"

        # Check if top_3_docs contains relevant information
        assert len(top_3_docs) > 0, "top_3_docs should not be empty"
        print(f"Top 3 docs snippet: {top_3_docs}...")  # Print first 100 characters

        # Additional checks based on the specific question
        if "notochord" in question:
            assert (
                "list all genes by territory" in top_3_docs.lower()
            ), "Expected 'notochord' in list all genes by territory"
        if "Brachyury" in question:
            assert (
                "regulatory" in top_3_docs.lower()
            ), "Expected 'regulatory' in top_3_docs"
        if "larval stage" in question:
            assert "stage" in top_3_docs.lower(), "Expected 'stage' in top_3_docs"


if __name__ == "__main__":
    pytest.main([__file__])
