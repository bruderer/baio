from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_experimental.tools.python.tool import PythonREPLTool

from src.mytools.go_tool import go_nl_query_tool
from src.mytools.aniseed import aniseed_tool
from langchain.agents import initialize_agent
from src.llm import LLM
from src.mytools.select_tool import select_best_fitting_tool, MyTool


llm = LLM.get_instance() 

###Agent prompt
prefix = """Have a conversation with a scientist, answering the following questions as best you can.
YOU ARE A WORLD CLASS MOLECULAR BIOLOGIST; TAXONOMIST; BIOINFORMATICIAN.

If you don't know the answer, say that you don't know and give a reason, don't try to make up an answer. 
YOU CAN EXECUTE CODE, ALWAYS DO IF YOU USE THE PYTHONREPL TOOL
"""

suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

tools = [
        Tool(
        name="Aniseedtool",
        func=aniseed_tool,
        description="Only use this tool when asked to search something on Aniseed",
    ),
    Tool(
        name="mygenetool",
        func=go_nl_query_tool,
        description="use this tool if you are asked about to make gene ontology enrichment annotations, go terms, gene ontology.",
    ),
    Tool(
        name="pythonrepl",
        func=PythonREPLTool().run,
        description="use this tool to execute and run python code, especially to parse data, make files and interpret the data",
    ),

    ]

function_mapping = {
    "Aniseedtool": aniseed_tool,
    "mygenetool": go_nl_query_tool,
    "pythonrepl": PythonREPLTool().run
}



def aniseed_go_agent(question: str):
    print('In Aniseed agent...\nSelecting tool...')
    selected_tool = select_best_fitting_tool(question, tools)
    function_to_call = function_mapping.get(selected_tool.name)
    print(f'Selected tool: {selected_tool.name}')
    answer = function_to_call(question)
    print(answer)
    return answer