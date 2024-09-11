import os
import uuid

import pandas as pd
from langchain.chains import create_extraction_chain

from baio.src.non_llm_tools import GoFormater
from baio.src.non_llm_tools.utilities import Utils, log_question_uuid_json


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
                "gene_names": {
                    "type": "string",
                    "description": "Gene names that you find, always extrac ALL gene "
                    "names",
                },
            },
            "required": ["gene_names"],
        }

        # Input
        # Run chain
        chain = create_extraction_chain(schema, llm)
        result = chain.run(self.natural_language_string)
        print(result)
        # unpacking the list and splitting
        gene_list = [gene_dict["gene_names"].strip() for gene_dict in result]

        return gene_list


def go_nl_query_tool(nl_input: str, llm, embedding) -> pd.DataFrame:
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

    question_uuid = str(uuid.uuid4())

    try:
        extractor = NaturalLanguageExtractors(nl_input)
        gene_list = extractor.gene_name_extractor(llm)

        gof = GoFormater(gene_list)
        final_go_df = gof.go_gene_annotation()
        final_go_df = Utils.parse_refseq_id(final_go_df)

        base_dir = os.getcwd()
        output_dir = os.path.join(base_dir, "baio", "data", "output", "gene_ontology")
        os.makedirs(output_dir, exist_ok=True)

        # Save to the original path (will be overwritten each time)
        original_file_name = "go_annotation.csv"
        original_file_path = os.path.join(output_dir, original_file_name)
        final_go_df.to_csv(original_file_path, index=False)

        # Save to the UUID-based path
        uuid_file_name = f"go_annotation_{question_uuid}.csv"
        uuid_file_path = os.path.join(output_dir, uuid_file_name)
        final_go_df.to_csv(uuid_file_path, index=False)

        # Log the query using the standard logging function
        log_file_path = os.path.join(output_dir, "go_nl_query_log.json")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        full_url = "N/A"  # There's no API call for this tool, so we use N/A
        log_question_uuid_json(
            question_uuid,
            nl_input,
            uuid_file_name,
            output_dir,
            log_file_path,
            full_url,
            answer=str(final_go_df.head()),
            tool="go_nl_query",
        )

        return final_go_df.head()

    except Exception as e:
        print(f"Error in GO NL Query Tool: {str(e)}")
        raise


def nl_gene_protein_name_tool(nl_input: str, llm, embedding) -> str:
    """
    Used when the input is a natural language written query containing gene or protein
    names.
    This tool extracts gene and protein names from the input.

    Parameters:
    nl_input (str): A natural language string containing gene or protein names to be processed

    Returns:
    str: A string containing the extracted gene and protein names
    """
    question_uuid = str(uuid.uuid4())
    try:
        prompt = "You are a world leading molecular biologist and geneticist. "
        "You know everything about proteins and genes and you are an expert in gene and"
        " protein names. Please extract the gene and protein names from the following. "
        "ALWAYS ANSWER 'no gene' IF THERE ARE NO GENE NAMES IN THE SENTENCE. "
        "sentence: "
        extractor = NaturalLanguageExtractors(prompt + nl_input)
        gene_protein_list = extractor.gene_name_extractor(llm)

        # Convert the list to a comma-separated string
        gene_protein_names = ", ".join(gene_protein_list)
        if gene_protein_names == "":
            gene_protein_names = "None"
        base_dir = os.getcwd()
        output_dir = os.path.join(
            base_dir, "baio", "data", "output", "gene_protein_names"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save to the original path (will be overwritten each time)
        original_file_name = "gene_protein_names.txt"
        original_file_path = os.path.join(output_dir, original_file_name)
        with open(original_file_path, "w") as f:
            f.write(gene_protein_names)

        # Save to the UUID-based path
        uuid_file_name = f"gene_protein_names_{question_uuid}.txt"
        uuid_file_path = os.path.join(output_dir, uuid_file_name)
        with open(uuid_file_path, "w") as f:
            f.write(gene_protein_names)

        # Log the query using the standard logging function
        log_file_path = os.path.join(output_dir, "nl_gene_protein_name_log.json")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        full_url = "N/A"  # There's no API call for this tool, so we use N/A
        log_question_uuid_json(
            question_uuid,
            nl_input,
            uuid_file_name,
            output_dir,
            log_file_path,
            full_url,
            answer=gene_protein_names,
            tool="nl_gene_protein_name",
        )

        return gene_protein_names

    except Exception as e:
        print(f"Error in NL Gene/Protein Name Tool: {str(e)}")
        raise
