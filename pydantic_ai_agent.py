"""
Experimental version of the research agent using PydanticAI instead of manual tool-calling loop.
The manual version is the primary implementation. It shows an understanding of how it all works.
But this version is to show how to automate it all. PydanticAI handles the tool-call loop internally,
and `output_type=CompanyReport` replaces our entire manual format_report() + retry + validation logic.
"""

import sys
from dotenv import load_dotenv
from pydantic_ai import Agent
from tools import search_web, search_news, fetch_webpage
from models import CompanyReport

load_dotenv()

visited_urls = []
agent = Agent(
    "groq:llama-3.3-70b-versatile",
    output_type=CompanyReport,
    system_prompt=(
        """
        You are a research agent. Use the search_web_tool, search_news_tool and 
        fetch_webpage_tool tools to gather real, current information 
        before answering. Always search first, then fetch at least one 
        relevant page - search snippets alone are too shallow for a good report. 
        Base your answer only on what the tools return - do not invent facts.
        """
    )
)

@agent.tool_plain
def search_web_tool(query: str) -> str:
    """Search the web for information on a topic."""
    return search_web(query)

@agent.tool_plain
def search_news_tool(query: str) -> str:
    """Search specifically for recent news articles about a topic."""
    return search_news(query)

@agent.tool_plain
def fetch_webpage_tool(url: str) -> str:
    """Fetch and read the full text content of one specific webpage URL."""
    result = fetch_webpage(url)
    if not result.startswith("Could not fetch"):
        visited_urls.append(url)
    return result

def main():
    visited_urls.clear()

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("What would you like me to research? ")
    print(f"\nResearching (PydanticAI version): {query}\n")

    result = agent.run_sync(query)
    report = result.output
    report.sources = visited_urls

    print("\nFINAL REPORT (PydanticAI): \n")
    print(f"Company: {report.company}")
    print(f"Overview: {report.overview}")
    print(f"Headquarters: {report.headquarters}")
    print(f"Founded: {report.founded}")
    print(f"Products/Services: {', '.join(report.product_services)}")
    print(f"Sources: {', '.join(report.sources)}")

if __name__ == "__main__":
    main()