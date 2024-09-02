import uuid
from typing import Optional

from pydantic import BaseModel, Field


class BaseANISEEDQuery(BaseModel):
    full_url: Optional[str] = Field(
        default="",
        description="The full URL of the API request (to be filled automatically)",
    )
    question_uuid: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question",
    )


class AllGenesQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_genes", description="This is the required function"
    )
    organism_id: int = Field(..., description="The ID of the organism to query")
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


class AllGenesByStageQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_genes_by_stage", description="This is the required function"
    )
    organism_id: int = Field(..., description="The ID of the organism to query")
    stage: str = Field(..., description="The specific developmental stage to query")
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


class AllGenesByStageRangeQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_genes_by_stage_range", description="This is the required function"
    )
    organism_id: int = Field(..., description="The ID of the organism to query")
    start_stage: str = Field(
        ..., description="The starting developmental stage for the range query"
    )
    end_stage: str = Field(
        ..., description="The ending developmental stage for the range query"
    )
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


class AllGenesByTerritoryQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_genes_by_territory", description="This is the required function"
    )
    organism_id: int = Field(..., description="The ID of the organism to query")
    cell: str = Field(..., description="The specific cell or territory to query")
    search: Optional[str] = Field(
        None,
        description="Optional search term to filter results, leave empty if not needed",
    )


class AllTerritoriesByGeneQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_territories_by_gene", description="This is the required function"
    )
    organism_id: int = Field(..., description="The ID of the organism to query")
    gene: str = Field(..., description="The gene to query")
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


class AllClonesByGeneQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_clones_by_gene", description="This is the required function"
    )
    organism_id: int = Field(..., description="The ID of the organism to query")
    gene: str = Field(..., description="The gene to query")
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


class AllConstructsQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_constructs", description="This is the required function"
    )
    organism_id: int = Field(..., description="The ID of the organism to query")
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


class AllMolecularToolsQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_molecular_tools", description="This is the required function"
    )
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


class AllPublicationsQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_publications", description="This is the required function"
    )
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


class AllRegulatoryRegionsQuery(BaseANISEEDQuery):
    required_function: str = Field(
        default="all_regulatory_regions", description="This is the required function"
    )
    organism_id: int = Field(..., description="The ID of the organism to query")
    search: Optional[str] = Field(
        None, description="Optional search term to filter results"
    )


ANISEEDQueryRequest = (
    AllGenesQuery
    | AllGenesByStageQuery
    | AllGenesByStageRangeQuery
    | AllGenesByTerritoryQuery
    | AllTerritoriesByGeneQuery
    | AllClonesByGeneQuery
    | AllConstructsQuery
    | AllMolecularToolsQuery
    | AllPublicationsQuery
    | AllRegulatoryRegionsQuery
)


def get_query_class(function: str):
    query_class_map = {
        "all_genes": AllGenesQuery,
        "all_genes_by_stage": AllGenesByStageQuery,
        "all_genes_by_stage_range": AllGenesByStageRangeQuery,
        "all_genes_by_territory": AllGenesByTerritoryQuery,
        "all_territories_by_gene": AllTerritoriesByGeneQuery,
        "all_clones_by_gene": AllClonesByGeneQuery,
        "all_constructs": AllConstructsQuery,
        "all_molecular_tools": AllMolecularToolsQuery,
        "all_publications": AllPublicationsQuery,
        "all_regulatory_regions": AllRegulatoryRegionsQuery,
    }
    return query_class_map.get(function, ANISEEDQueryRequest)
