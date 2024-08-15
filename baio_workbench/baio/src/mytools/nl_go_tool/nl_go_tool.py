import os

import pandas as pd
from langchain.chains import create_extraction_chain

from baio.src.non_llm_tools import GoFormater
from baio.src.non_llm_tools.utilities import Utils


class NaturalLanguageExtractors:
    """This class contains methods to extract certain information in a structured manner
    from natural language."""

    def __init__(self, natural_language_string):
        """
        Initialize the extractor with the natural_language_string input by the user.

        Parameters:
        natural_language_string (str): The string from which the information has to be
        extracted.
        """
        self.natural_language_string = natural_language_string

    def gene_name_extractor(self, llm) -> list:
        """
        Extracts gene names and returns a list.
        """
        schema = {
            "properties": {
                "gene_names": {"type": "string"},
            },
            "required": ["gene_names"],
        }

        # Input
        # Run chain
        chain = create_extraction_chain(schema, llm)
        result = chain.run(self.natural_language_string)

        # unpacking the list and splitting
        gene_list = [gene.strip() for gene in [result[0]["gene_names"]][0].split(",")]
        return gene_list


def go_nl_query_tool(nl_input: str, llm) -> pd.DataFrame:
    """Used when the input is a natural language written query containing gene names
    that need a GO annotation.
    Tool to find gene ontologies (using mygene), outputs data frame with GO & gene id
    annotated gene names

    Parameters:
    input_string (str): A natural language string containing gene names that have to be
    processed

    Returns:
    final_dataframe (dataframe): A df containing annotated genes with GO & IDs from
    mygene.
    """

    # we extract all the go terms and ids for all genes in this list

    extractor = NaturalLanguageExtractors(nl_input)
    gene_list = extractor.gene_name_extractor(llm)
    gof = GoFormater(gene_list)
    final_go_df = gof.go_gene_annotation()
    final_go_df = Utils.parse_refseq_id(final_go_df)
    base_dir = os.getcwd()
    SAVE_PATH = os.path.join(
        base_dir, "baio", "data", "output", "gene_ontology", "go_annotation.csv"
    )

    final_go_df.to_csv(SAVE_PATH)
    return final_go_df.head()
