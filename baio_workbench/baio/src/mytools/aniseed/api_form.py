import uuid
from typing import Optional

from pydantic import BaseModel, Field


class ANISEEDQueryRequest(BaseModel):
    required_function: str = Field(
        default="the required function",
        description="given the question, what function do you need to call to answer "
        "it?",
    )
    parameter_1_name: str = Field(
        default="To be filled",
        description="Dependant on the function you have to choose the appropriate"
        "parameter1 to answer the question",
    )
    parameter_1_value: str = Field(
        default="To be filled",
        description="Dependant on the function you have to choose the appropriate value"
        " for  parameter1 to answer the question",
    )
    parameter_2_name: str = Field(
        default="To be filled",
        description="Dependant on the function you have to choose the appropriate"
        "parameter2 to answer the question",
    )
    parameter_2_value: str = Field(
        default="To be filled",
        description="Dependant on the function you have to choose the appropriate value"
        " for  parameter2 to answer the question",
    )
    parameter_3_name: str = Field(
        default="To be filled",
        description="Dependant on the function you have to choose the appropriate "
        " parameter3 to answer the question",
    )
    parameter_3_value: str = Field(
        default="To be filled",
        description="Dependant on the function you have to choose the appropriate value"
        " for  parameter3 to answer the question",
    )
    full_url: Optional[str] = Field(default="", description="")
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )
