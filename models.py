from pydantic import BaseModel, HttpUrl
from typing import List

class CompanyReport(BaseModel):
    company: str
    overview: str
    product_services: List[str]
    headquarters: str|None=None
    founded: str|None=None
    sources: List[HttpUrl]