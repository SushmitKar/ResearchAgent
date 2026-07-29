import os
import json
from groq import Groq
from tools import search_web, search_news, fetch_webpage
from models import CompanyReport

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# MODEL = "llama-3.3-70b-versatile"
MODEL = "llama-3.1-8b-instant"
# MODEL = "openai/gpt-oss-120b"

MAX_ITERATIONS = 6
MAX_FORMAT_RETRIES = 2
MAX_LLM_RETRIES = 3

AVAILABLE_TOOLS = {
    "search_web": search_web,
    "search_news": search_news,
    "fetch_webpage": fetch_webpage,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a topic. Returns titles, URLs, and short content snippets from top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'LumiQ company overview'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search specifically for recent news articles about a topic. Use this in addition to search_web when the user wants recent developments, not just general company info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The news search query, e.g. 'LumiQ recent funding'"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch and read the full text content of one specific webpage URL. Use this after search_web has found a relevant URL you want to read in more depth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the webpage to read, e.g. 'https://example.com'",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

GATHER_SYSTEM_PROMPT = """You are a research agent. Given a topic, use the search_web, search_news,
and fetch_webpage tools to gather real, current information before answering.
Always call search_web first for general information. Optionally call search_news
if recent developments would be useful. You MUST call fetch_webpage on at least
one relevant result before finishing - search snippets alone are too shallow for
a real report. Prefer fetching 2 different pages if the topic supports it, to
get enough depth for a detailed overview.

Once you have enough information, STOP calling tools and write a plain-text
summary of everything you found: overview, products/services, headquarters,
founding year, and any other concrete facts. Do not format this as JSON -
just write clear notes in plain English. A separate step will handle formatting.

Do not make up facts you have not found through the tools."""


def call_llm(messages, tools=None):
    """Wraps the Groq call with retries. The model occasionally emits a
    malformed tool call which raises an error. Retrying the same request
    usually succeeds since the failure is probabilistic. Returns None if
    every attempt fails."""
    kwargs = {"model": MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    for attempt in range(MAX_LLM_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"[agent] LLM call failed (attempt {attempt + 1}/{MAX_LLM_RETRIES}): {e}")

    return None


def gather_research(user_query: str):
    """Let the LLM use tools to research. Returns (notes_text, sources_list)."""
    messages = [
        {"role": "system", "content": GATHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    visited_urls = []
    nudged_for_source = False

    for iteration in range(MAX_ITERATIONS):
        response = call_llm(messages, tools=TOOL_SCHEMAS)

        if response is None:
            print("[agent] giving up: Groq kept failing to generate a valid tool call")
            return None, visited_urls

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            if not visited_urls and not nudged_for_source:
                print("[agent] model tried to finish without fetching a page - nudging it to fetch one")
                nudged_for_source = True
                messages.append(assistant_message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You haven't called fetch_webpage yet. Pick the most relevant "
                            "URL from your search results and fetch it before giving your "
                            "final summary."
                        ),
                    }
                )
                continue
            return assistant_message.content, visited_urls

        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"[agent] calling {tool_name}({tool_args})")

            tool_function = AVAILABLE_TOOLS[tool_name]
            result = tool_function(**tool_args)

            if tool_name == "fetch_webpage" and not result.startswith("Could not fetch"):
                visited_urls.append(tool_args["url"])

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": result}
            )

    return None, visited_urls  # hit MAX_ITERATIONS without finishing


def build_example_json():
    """Turns the Pydantic schema into a filled EXAMPLE instance (not the raw schema),
    so the model has something to imitate rather than something to echo back."""
    properties = CompanyReport.model_json_schema()["properties"]
    example = {}
    for field, spec in properties.items():
        if field == "sources":
            example[field] = []
        elif spec.get("type") == "array":
            example[field] = [spec.get("description", "value")]
        else:
            example[field] = spec.get("description", "value")
    return example


def format_report(research_notes: str, visited_urls: list):
    """Turn plain-text notes into validated structured JSON."""
    example_str = json.dumps(build_example_json(), indent=2)

    FORMAT_SYSTEM_PROMPT = (
        "You convert research notes into structured JSON.\n"
        "Here is an EXAMPLE of the exact JSON shape to produce "
        "(replace every value with real information from the notes):\n\n"
        f"{example_str}\n\n"
        "Rules:\n"
        "- Return ONLY a JSON object with these exact keys, filled with real values.\n"
        "- No markdown, no code fences, no explanation, no placeholder text.\n"
        "- Leave 'sources' as an empty array - it is filled in separately.\n"
        "- If a field is genuinely unknown, use an empty string or empty list."
    )

    format_messages = [
        {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research notes:\n\n{research_notes}"},
    ]

    for attempt in range(MAX_FORMAT_RETRIES):
        response = call_llm(format_messages)

        if response is None:
            print("[agent] giving up formatting: Groq kept failing to respond")
            return None

        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            report = CompanyReport.model_validate_json(raw)
            report.sources = visited_urls  # Python owns this, not the LLM
            return report
        except Exception as e:
            print(f"[agent] validation failed (attempt {attempt + 1}): {e}")
            format_messages.append({"role": "assistant", "content": raw})
            format_messages.append(
                {"role": "user", "content": f"That was invalid: {e}. Return ONLY corrected JSON."}
            )

    return None


def run_agent(user_query: str):
    research_notes, visited_urls = gather_research(user_query)

    if research_notes is None:
        print("[agent] stopped: reached max iterations during research")
        return None

    return format_report(research_notes, visited_urls)