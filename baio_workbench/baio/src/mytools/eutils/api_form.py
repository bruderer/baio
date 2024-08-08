import uuid
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class EutilsAPIRequest(BaseModel):
    url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        description=(
            "URL endpoint for the NCBI Eutils API, always use esearch except "
            "for db=snp, then use "
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi."
        ),
    )
    method: str = Field(
        default="GET",
        description="HTTP method for the request. Typically 'GET' or 'POST'.",
    )
    headers: Dict[str, str] = Field(
        default={"Content-Type": "application/json"},
        description="HTTP headers for the request. Default is JSON content type.",
    )
    db: str = Field(
        ...,
        description="Database to search. E.g., 'gene' for gene database, 'snp' for SNPs"
        ",'omim' for genetic diseases. ONLY ONE to best answer the question",
    )
    retmax: int = Field(..., description="Maximum number of records to return.")
    retmode: str = Field(
        default="json",
        description="Return mode, determines the format of the response. Commonly "
        "'json' or 'xml'.",
    )
    sort: Optional[str] = Field(
        default="relevance",
        description="Sorting parameter. Defines how results are sorted.",
    )
    term: Optional[str] = Field(
        None,
        description="Search term. Used to query the database. if it is for a SNP always"
        "remove rs before the number",
    )
    id: Optional[int] = Field(
        None,
        description="ONLY for db=snp!!! Identifier(s) in the search query for specific"
        " records when looking for SNPs. Obligated integer without the 'rs' prefix, use"
        "user question to fill.",
    )
    response_format: str = Field(
        default="json",
        description="Expected format of the response, such as 'json' or 'xml'.",
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )
    full_search_url: Optional[str] = Field(
        default="TBF",
        description="Search url for the first API call -> obtian id's for call n2",
    )


class EfetchRequest(BaseModel):
    url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        description="URL endpoint for the NCBI Efetch API.",
    )
    db: str = Field(
        ..., description="Database to fetch from. E.g., 'gene' for gene database."
    )
    id: Union[int, str, List[Union[int, str]]] = Field(
        ..., description="Comma-separated list of NCBI record identifiers."
    )
    retmode: str = Field(
        default="xml",
        description="Return mode, determines the format of the response. Commonly 'xml'"
        " or 'json'.",
    )
    full_search_url: Optional[str] = Field(
        default="TBF", description="Search url for the efetch API call"
    )
