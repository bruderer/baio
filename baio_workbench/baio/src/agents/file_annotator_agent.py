# from langchain.agents import AgentType, Tool, ZeroShotAgent, initialize_agent
# from langchain_experimental.tools.python.tool import PythonREPLTool

# from baio.src.non_llm_tools import go_file_tool

# prefix = """Have a conversation with a scientist, answering the following questions as
# best you can.
# YOU ARE A WORLD CLASS MOLECULAR BIOLOGIST; TAXONOMIST; BIOINFORMATICIAN.
# If you don't know the answer, say that you don't know and give a reason, don't try to
# make up an answer.

# You are the best at annotating local files with GO terms and allow the scientist to
# explore the files he provides you
# """

# suffix = """Begin!"

# {chat_history}
# Question: {input}
# {agent_scratchpad}"""

# file_annotator_tools = [
#     Tool(
#         name="mygenetool_file input",
#         func=go_file_tool,
#         description="Use this tool when you are given a file or path to annotate its"
#         "genes with their gene ontology, always input file and the gene column",
#     ),
#     Tool(
#         name="pythonrepl",
#         func=PythonREPLTool().run,
#         description="use this tool to execute and run python code",
#     ),
# ]

# file_annotator_promt = ZeroShotAgent.create_prompt(
#     file_annotator_tools,
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=["input", "chat_history", "agent_scratchpad"],
# )


# def file_annotator_agent(llm, memory):
#     return initialize_agent(
#         file_annotator_tools,
#         llm,
#         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#         verbose=True,
#         memory=memory,
#     )
