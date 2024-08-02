from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from src.mytools.select_tool import select_best_fitting_tool, MyTool
from src.mytools.eutils_tool import eutils_tool
from src.mytools.BLAST_structured_output import blast_tool
from src.mytools.BLAT_structured_output import BLAT_tool
from src.llm import LLM
import os
from langchain.chat_models import ChatOpenAI

llm = LLM.get_instance()


tools = [
        MyTool(
        name="eutils_tool",
        func=eutils_tool,
        description="Always use this tool when you are making requests on NCBI except when you are given a DNA or protein sequence",
    ),
        MyTool(
        name="blast_tool",
        func=blast_tool,
        description="With this tool you have access to the BLAST data base on NCBI, use it for queries about a DNA or protein sequence\
        EXCEPT if the question is about aligning a sequence with a specifice organisms, then use BLAT_tool.",
    ),
    MyTool(
        name="BLAT_tool",
        func=BLAT_tool,
        description="Use for questions such s 'Align the DNA sequence to the human:ATTCGCC...; If you are asked to\
            With this tool you have access to the ucsc genome data base. It can find where DNA sequences are aligned on the organisms genome, exact positions etc. ",
),
    ]

function_mapping = {
    "eutils_tool": eutils_tool,
    "blast_tool": blast_tool,
    "BLAT_tool": BLAT_tool
}

def ncbi_agent(question: str):
    print('In NCBI agent...\nSelecting tool...')
    selected_tool = select_best_fitting_tool(question, tools)
    function_to_call = function_mapping.get(selected_tool.name)
    print(f'Selected tool: {selected_tool.name}')
    answer = function_to_call(question)
    print(answer)
    return answer
