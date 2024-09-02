import uuid
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class BLATQueryRequest(BaseModel):
    url: str = Field(
        default="https://genome.ucsc.edu/cgi-bin/hgBlat?",
        description="ALWAYS USE DEFAULT",
    )
    query: Optional[str] = Field(
        None,
        description="Nucleotide or protein sequence for the BLAT query, make sure to "
        "always keep the entire sequence given.",
    )
    ucsc_db: str = Field(
        default="hg38",
        description="Genome assembly to use in the UCSC Genome Browser, use the correct"
        "db for the organisms. Human:hsg38; Mouse:mm10; Dog:canFam6",
    )
    # Additional fields for UCSC Genome Browser
    ucsc_track: str = Field(
        default="genes",
        description="Genome Browser track to use, e.g., 'genes', 'gcPercent'.",
    )
    ucsc_region: Optional[str] = Field(
        None,
        description="Region of interest in the genome, e.g., 'chr1:100000-200000'.",
    )
    ucsc_output_format: str = Field(
        default="json",
        description="Output format for the UCSC Genome Browser, e.g., 'bed', 'fasta'.",
    )
    ucsc_query_type: str = Field(
        default="DNA",
        description="depends on the query DNA, protein, translated RNA, or translated "
        "DNA",
    )
    full_url: str = Field(
        default="TBF",
        description="Url for the BLAT query, use the given examples to make the "
        "according one!",
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )
    retmode: str = Field(
        default="json",
        description="Return mode for the API response, e.g., 'json', 'xml', 'text'.",
    )


class BLATdb(BaseModel):
    ucsc_db: str = Field(
        default="hg38",
        description="Genome assembly to use in the UCSC Genome Browser, use the correct"
        " db for the organisms. Human:hsg38; Mouse:mm10; Dog:canFam6",
    )
