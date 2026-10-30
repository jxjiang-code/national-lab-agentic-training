import os

from pydantic import BaseModel
from pydantic import EmailStr
from pydantic import Field

from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching


class PaperClaim(BaseModel):
    claim: str
    evidence: str

class Paper(BaseModel):
    title: str
    authors: list[str]
    venue: str | None
    contacts: list[EmailStr]
    year: int = Field(ge=1888, le=2100) | None
    claims: list[PaperClaim]

def main(model: str, paper: str):
    with open(paper) as fp:
        paper_text = fp.read()
        
    agent = Agent(model, output_type=Paper, capabilities=[Caching()])
    result = agent.run_sync('Extract the claims and evidence from this paper:\n {paper_text}')

if __name__ == '__main__':
    main(model, paper)