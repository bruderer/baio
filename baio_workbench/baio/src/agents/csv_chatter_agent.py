import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

PATH = "./baio/data/output/aniseed/aniseed_out.csv"
filechatter_instructions = """Have a conversation with a scientist, answering the
following questions as best you can.
YOU ARE A WORLD CLASS MOLECULAR BIOLOGIST; TAXONOMIST; BIOINFORMATICIAN.
You know everything about data hadling, especially pandas and you love to use python for
it. If you don't know the answer, say that you don't know and give a reason, don't try
to make up an answer.
You Give precise and good answers consering the data in the panda dataframe you are
given, you write python code to get this information.
Place all the genreted files into this directory: './baio/data/output/csv_chatter'.
USE baio NOT bio answer the follwoing question:
"""


def csv_chatter_agent(question, file_paths: list, llm):
    valid_file_paths = [fp for fp in file_paths if fp is not None]
    if valid_file_paths:
        csv_agent = csv_agent_creator(valid_file_paths, llm)
        return csv_agent.run(filechatter_instructions + question)
    else:
        return None


def csv_agent_creator(path, llm):
    """Input: path to csv, will be loaded as panda and input in pandas_dataframe_agent
    when it is initiated

    RETURN: agent
    """

    df1 = pd.read_csv(path[0])
    print(path)
    if path and len(path) > 1 and path[1] != "":
        df2 = pd.read_csv(path[1])
        csv_chatter_agent = create_pandas_dataframe_agent(
            llm,
            [df1, df2],
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        return csv_chatter_agent
    else:
        csv_chatter_agent = create_pandas_dataframe_agent(
            llm,
            df=df1,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        return csv_chatter_agent
