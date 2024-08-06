import uuid
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class BlastQueryRequest(BaseModel):
    url: str = Field(
        default="https://blast.ncbi.nlm.nih.gov/Blast.cgi?",
        description="ALWAYS USE DEFAULT, DO NOT CHANGE",
    )
    cmd: str = Field(
        default="Put",
        description="Command to execute, 'Put' for submitting query, 'Get' for "
        "retrieving results.",
    )
    program: Optional[str] = Field(
        default="blastn",
        description="BLAST program to use, e.g., 'blastn' for nucleotide BLAST.",
    )
    database: str = Field(
        default="nt",
        description="Database to search, e.g., 'nt' for nucleotide database.",
    )
    query: Optional[str] = Field(
        None,
        description="Nucleotide or protein sequence for the BLAST or blat query, make "
        "sure to always keep the entire sequence given.",
    )
    format_type: Optional[str] = Field(
        default="Text", description="Format of the BLAST results, e.g., 'Text', 'XML'."
    )
    rid: Optional[str] = Field(
        None, description="Request ID for retrieving BLAST results."
    )
    other_params: Optional[dict] = Field(
        default={"email": "noah.bruderer@uib.no"},
        description="Other optional BLAST parameters, including user email.",
    )
    max_hits: int = Field(
        default=15, description="Maximum number of hits to return in the BLAST results."
    )
    sort_by: Optional[str] = Field(
        default="score",
        description="Criterion to sort BLAST results by, e.g., 'score', 'evalue'.",
    )
    megablast: str = Field(
        default="on", description="Set to 'on' for human genome alignemnts"
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )
    full_url: Optional[str] = Field(
        default="TBF", description="Url used for the blast query"
    )
