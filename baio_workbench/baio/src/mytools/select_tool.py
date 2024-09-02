from typing import Callable, Optional

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from baio.src.llm import LLM

llm = LLM.get_instance()


class MyTool(BaseModel):
    name: str = Field(default="")
    func: Optional[Callable[..., str]] = Field(
        default=None,
    )
    description: str = Field(default="", description="The description of the tool")


class ToolSelector(BaseModel):
    name: str = Field(
        default="",
        description="The name of the best fitting tool to answer the question",
    )
    description: str = Field(
        default="", description="The description of the best fitting tool tool"
    )


def select_best_fitting_tool(question: str, tools: list, llm):
    """FUNCTION to select tool to answer user questions"""
    tool_structured_output_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in "
                "structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input: "
                "{input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    runnable = create_structured_output_runnable(
        ToolSelector, llm, tool_structured_output_prompt
    )
    # retrieve relevant info to question
    # keep top 3 hits
    selected_tool = runnable.invoke({"input": f"{question} based on {tools}"})
    print(f"\033[92mselected tool is: {selected_tool}\033[0m")
    return selected_tool
