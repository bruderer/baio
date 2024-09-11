from baio.src.mytools import (  # nl_gene_protein_name_tool,
    BLAT_tool,
    MyTool,
    blast_tool,
    eutils_tool,
    go_nl_query_tool,
    select_best_fitting_tool,
)

tools = [
    MyTool(
        name="eutils_tool",
        func=eutils_tool,
        description="Always use this tool when you are making requests on NCBI except "
        "when you are given a DNA or protein sequence",
    ),
    MyTool(
        name="mygenetool",
        func=go_nl_query_tool,
        description="use this tool if you are asked about finding gene ontology."
        "enrichment annotations, go terms, gene ontology.",
    ),
    # MyTool(
    #     name="gene_protein_name_tool",
    #     func=nl_gene_protein_name_tool,
    #     description="use this tool if you are asked to find gene and protein names in a"
    #     " sentence. NEVER USE IT IF YOU ARE ASKED FOR A GENE ALIAS OR THE OFFICIAL "
    #     "GENE SYMBOL OF A GENE, ONLY USE IT IF YOU ARE EXPLICITLY ASKED TO EXTRACT "
    #     "A GENE NAME OR PROTEIN NAME OUT OF A SENTENCE OR TEXT!!!!",
    # ),
    MyTool(
        name="blast_tool",
        func=blast_tool,
        description="With this tool you have access to the BLAST data base on NCBI, use"
        "it for queries about a DNA or protein sequence\
        EXCEPT if the question is about aligning a sequence with a specifice organisms,"
        "then use BLAT_tool.",
    ),
    MyTool(
        name="BLAT_tool",
        func=BLAT_tool,
        description="Use for questions such s 'Align the DNA sequence to the "
        "human:ATTCGCC...; or If you are asked to identify on what chromosome a DNA\
        Sequence is located.\
        With this tool you have access to the ucsc genome data base. It can find where"
        " DNA sequences are aligned on the organisms genome, exact positions etc. ",
    ),
]

function_mapping = {
    "eutils_tool": eutils_tool,
    "blast_tool": blast_tool,
    "BLAT_tool": BLAT_tool,
    "mygenetool": go_nl_query_tool,
    # "gene_protein_name_tool": nl_gene_protein_name_tool,
}


def baio_agent(question: str, llm, embedding):
    print("In Baio agent...\nSelecting tool...")
    selected_tool = select_best_fitting_tool(question, tools, llm)
    print(f"Selected tool: {selected_tool.name}")
    selected_tool = function_mapping.get(selected_tool.name)
    try:
        answer = selected_tool(question, llm, embedding)  # type: ignore
        print(answer)
        return answer
    except Exception as e:
        error_message = f"Error occurred: {str(e)}"
        print(error_message)
        return error_message
