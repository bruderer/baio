from langchain.tools import Tool

from baio.src.mytools.aniseed import aniseed_tool
from baio.src.mytools.select_tool import select_best_fitting_tool

# from langchain_experimental.tools.python.tool import PythonREPLTool


# Agent prompt
prefix = """Have a conversation with a scientist, answering the following questions as
best you can. YOU ARE A WORLD CLASS MOLECULAR BIOLOGIST; TAXONOMIST; BIOINFORMATICIAN.

If you don't know the answer, say that you don't know and give a reason, don't try to
make up an answer. YOU CAN EXECUTE CODE, ALWAYS DO IF YOU USE THE PYTHONREPL TOOL
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
    # Tool(
    #     name="pythonrepl",
    #     func=PythonREPLTool().run,
    #     description="use this tool to execute and run python code, especially to "
    #     "parse"
    #     "data, make files and interpret the data",
    # ),
]

function_mapping = {
    "Aniseedtool": aniseed_tool,
    # "pythonrepl": PythonREPLTool().run,
}


def aniseed_agent(question: str, llm):
    print("In Aniseed agent...\nSelecting tool...")
    selected_tool = select_best_fitting_tool(question, tools, llm)
    function_to_call = function_mapping.get(selected_tool.name)
    print(f"Selected tool: {selected_tool.name}")
    if selected_tool.name == "pythonrepl":
        answer = function_to_call(question, llm)  # type: ignore
    else:
        answer = function_to_call(question, llm)  # type: ignore
    print(answer)
    return answer
