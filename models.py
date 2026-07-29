from pydantic import BaseModel, Field
from typing import List, Optional


class CompanyReport(BaseModel):
    company: str = Field(description="Name of the company researched")
    overview: str = Field(description="A detailed 4-6 sentence overview covering what the company does, its main products or business model, market position, and any notable scale or recent developments")
    product_services: List[str] = Field(description="List of products or services offered")
    headquarters: Optional[str] = Field(default=None, description="Headquarters location, if known")
    founded: Optional[str] = Field(default=None, description="Founding year, if known")
    sources: List[str] = Field(default_factory=list, description="URLs actually used - filled in by Python, not the LLM")