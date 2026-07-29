import os
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

def search_web(query: str)->str:
    """
    Searches the web for a given query and returns a compact text summary
    of the top results, ready to feed back into an LLM.
    """
    results = tavily_client.search(query=query, max_results=5)
    formatted = []
    for r in results["results"]:
        formatted.append(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n"
        )
    return "\n---\n".join(formatted)

def search_news(query: str) -> str:
    """
    Searches for recent news articles for a topic and returns a compact text summary
    of the top results.
    """
    results = tavily_client.search(query=query, topic="news",max_results=5)

    formatted = []
    for r in results['results']:
        formatted.append(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n"
        )
    return "\n---\n".join(formatted)

def fetch_webpage(url: str) -> str:
    """
    Fetches a specific webpage and returns its main visible text content,
    trimmed to a reasonable length for the LLM to read.
    """
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Could not fetch {url}: {e}"

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)

    return text[:4000]