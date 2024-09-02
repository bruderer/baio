from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class AniseedStepDecider(BaseModel):
    """
    AniseedStepDecider _summary_

    Args:
        BaseModel (_type_): _description_
    """

    valid_functions = [
        "all_genes",
        "all_genes_by_stage",
        "all_genes_by_stage_range",
        "all_genes_by_territory",
        "all_territories_by_gene",
        "all_clones_by_gene",
        "all_constructs",
        "all_molecular_tools",
        "all_publications",
        "all_regulatory_regions",
    ]

    One_or_more_steps: bool = Field(
        default=False,
        description="Based on the documentation, do you require more than one API call "
        "to get the required information?",
    )
    functions_to_use_1: str = Field(
        default="all_genes",
        description="ONLY FUNCTION NAME:"
        "Write the function name you need to use to answer the question. "
        f"It can only be a function from this list: {valid_functions}",
    )
    functions_to_use_2: str = Field(
        default="all_genes",
        description="ONLY FUNCTION NAME:"
        "If more than one API call is required to answer the users "
        "question, write the second function name you need to use to answer the "
        f"question. It can only be a function from this list: {valid_functions}",
    )
    functions_to_use_3: str = Field(
        default="all_genes",
        description=f"ONLY FUNCTION NAME:"
        "If more than two API call are required to answer the users "
        "question, write the third function name you need to use to answer the "
        f"question. It can only be a function from this list: {valid_functions}",
    )


function_input = """
Here you have a list of functions used to retrieve information from the Aniseed
database.
def all_genes(self, organism_id, search=None):
    \"\"\"
    Returns a URL to list all genes for a given organism.
    Optionally, a search term can be provided to filter the genes.
    \"\"\"

def all_genes_by_stage(self, organism_id, stage, search=None):
    \"\"\"
    Returns a URL to list all genes for a given organism that are expressed at a
    specific stage.
    Optionally, a search term can be provided to filter the genes.
    \"\"\"

def all_genes_by_stage_range(self, organism_id, start_stage, end_stage, search=None):
    \"\"\"
    Returns a URL to list all genes for a given organism that are expressed between two
    stages.
    Optionally, a search term can be provided to filter the genes.
    \"\"\"

def all_genes_by_territory(self, organism_id, cell, search=None):
    \"\"\"
    Returns a URL to list all genes for a given organism that are expressed in a
    specific territory.
    Optionally, a search term can be provided to filter the genes.
    \"\"\"

def all_territories_by_gene(self, organism_id, gene, search=None):
    \"\"\"
    Returns a URL to list all territories where a specific gene is expressed for a
    given organism and in what stage.
    Optionally, a search term can be provided to filter the territories.
    \"\"\"

def all_clones_by_gene(self, organism_id, gene, search=None):
    \"\"\"
    Returns a URL to list all clones for a specific gene for a given organism.
    Optionally, a search term can be provided to filter the clones.
    \"\"\"

def all_constructs(self, organism_id, search=None):
    \"\"\"
    Returns a URL to list all constructs for a given organism.
    Optionally, a search term can be provided to filter the constructs.
    \"\"\"

def all_molecular_tools(self, search=None):
    \"\"\"
    Returns a URL to list all molecular tools in the database.
    Optionally, a search term can be provided to filter the tools.
    \"\"\"

def all_publications(self, search=None):
    \"\"\"
    Returns a URL to list all publications in the database.
    Optionally, a search term can be provided to filter the publications.
    \"\"\"

def all_regulatory_regions(self, organism_id, search=None):
    \"\"\"
    Returns a URL to list all regulatory regions for a given organism.
    Optionally, a search term can be provided to filter the regions.
    \"\"\"
"""


def ANISEED_multistep(question: str, llm):
    """FUNCTION to write api call for any ANISEED query,"""
    print("Finding the required aniseed api function to answer the question...\n")
    structured_output_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in "
                "structured formats.",
            ),
            (
                "human",
                "You have to decide what functions you need to answer the user question"
                "You have to make multiple API calls to answer certain user question "
                "and as an example, if you are asked to find genes"
                "that are expressed in the notochord of Ciona intestinalis in a stage"
                "range, you would first search all_genes_by_territory and then"
                "all_genes_by_stage_range, this will give all the information to filter"
                "later. VERY IMPORTANT: ONLY INPUT THE FUNCTION NAME "
                "Use the given format to extract information from the following input: "
                "or more API calls: {input}",
            ),
            (
                "human",
                "Tip: Make sure to answer in the correct format, make sure to respect ",
            ),
        ]
    )
    runnable = create_structured_output_runnable(
        AniseedStepDecider, llm, structured_output_prompt
    )

    one_or_more = runnable.invoke(
        {
            "input": f"to answer {question} do you need one or more Api calls to answer"
            f" it, base your answer on: {function_input}"
        }
    )
    if one_or_more.functions_to_use_1 not in one_or_more.valid_functions:
        one_or_more.functions_to_use_1 = None
    if one_or_more.functions_to_use_2 not in one_or_more.valid_functions:
        one_or_more.functions_to_use_2 = None
    if one_or_more.functions_to_use_3 not in one_or_more.valid_functions:
        one_or_more.functions_to_use_3 = None
    # retrieve relevant info to question

    return one_or_more
