# LumiQ Agentic AI Assignment — Company Research Agent

An autonomous research agent that takes a company name, decides which tools to
use, gathers real information from the web, and returns a structured,
validated report — built as a take-home assignment for LumiQ's Enterprise
AI / Agentic AI internship (Option A: Research Agent).

## Problem Statement

Enterprise AI systems rarely get to call an LLM once and trust the answer —
they need the model to actively go find current, real information, then
return it in a predictable shape other systems can consume. This project
demonstrates that pattern end-to-end: an agent that autonomously decides
when to search, when to read a specific page, and when it has enough
information to answer — then converts its findings into a validated
`CompanyReport` object instead of free-form text.

## Architecture

```
User query ("Research LumiQ")
        │
        ▼
 Phase 1 — Gather (agent.py: gather_research)
   LLM decides which tools to call, in a loop:
        │
        ├── search_web(query)      → Tavily search API
        └── fetch_webpage(url)     → requests + BeautifulSoup
        │
   Loop continues until the LLM has enough information
   and writes a plain-text research summary
        │
        ▼
 Phase 2 — Format (agent.py: format_report)
   A second, tool-free LLM call converts the plain-text
   summary into JSON matching the CompanyReport schema
        │
        ▼
 Pydantic validation (models.py: CompanyReport)
   Invalid JSON triggers an automatic retry with the
   validation error fed back to the model
        │
        ▼
 Python overrides `sources` with the real URLs it
 actually fetched (never trusts the LLM to report these)
        │
        ▼
Structured CompanyReport → printed + saved to report.md
```

## Key Design Decisions

**Two-phase design (gather, then format) instead of one combined call.**
Asking a model to decide on tool calls *and* produce strict JSON in the same
turn is unreliable — the two goals compete for the model's attention. Splitting
them into separate calls, each with one job, was the single biggest
reliability improvement made during development.

**Python owns the `sources` field, not the LLM.** Early testing showed the
model would occasionally invent a plausible-but-wrong URL (e.g. reporting
`lumiq.com` when the tool had actually fetched `lumiq.ai`). Every URL the
`fetch_webpage` tool successfully reads is tracked in Python and injected into
the final report afterward, regardless of what the model writes.

**Automatic retries on both LLM calls and JSON validation.** Groq's
function-calling occasionally emits a malformed tool call — a known
reliability characteristic of open-weight models, not a code bug. Since the
failure is probabilistic, retrying the identical request usually succeeds.
Separately, if the formatting step returns invalid JSON, the exact Pydantic
validation error is fed back to the model, which reliably self-corrects.

**Manual implementation as the primary submission.** The core agent
(`agent.py`) is built directly against the Groq API with no framework, so
every part of the tool-calling loop, retry logic, and validation is explicit
and fully understood — not delegated to a library.

## Bonus: PydanticAI Experimental Version

`pydantic_ai_agent.py` reimplements the same agent using
[PydanticAI](https://ai.pydantic.dev/), reusing the existing `tools.py` and
`models.py` unchanged. It demonstrates the same research task in roughly
40% of the code, since the framework absorbs the tool-call loop and the
JSON-validation-and-retry logic that `agent.py` implements by hand. It is a
secondary/comparison implementation, not a replacement for the primary one.

## Project Structure

```
lumiq-research-agent/
├── main.py                  # entry point for the manual agent
├── agent.py                 # manual agent: tool loop + structured formatting
├── tools.py                 # search_web() and fetch_webpage() tool functions
├── models.py                # CompanyReport Pydantic schema
├── pydantic_ai_agent.py     # bonus: same agent built with PydanticAI
├── requirements.txt
├── .env.example              # template for required API keys
├── report.md                 # generated after each run
└── README.md
```

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pydantic-ai        # only needed for the bonus version
   ```

3. Copy `.env.example` to `.env` and add your keys:
   ```
   GROQ_API_KEY=your_groq_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```
   - Groq API key: https://console.groq.com (free tier, no card required)
   - Tavily API key: https://tavily.com (free tier, no card required)

## Usage

```bash
# Manual implementation (primary)
python main.py "Research LumiQ"

# PydanticAI implementation (bonus)
python pydantic_ai_agent.py "Research LumiQ"
```

Each run prints the structured report to the console; the manual version
also saves a formatted `report.md`.

## Known Limitations

- **Model reliability:** Groq's Llama models occasionally produce malformed
  tool calls. This is mitigated with automatic retries but not eliminated —
  a run can still fail if all retries are exhausted.
- **No entity disambiguation:** the agent trusts whatever `search_web`
  returns and has no way to tell apart two different companies that share a
  similar name (discovered during testing: `lumiq.ai`, `lumiq.in`, and
  `lumiq.com` are not necessarily the same entity). Results should be
  spot-checked for well-known name collisions.
- **Fixed iteration cap:** `MAX_ITERATIONS` in `agent.py` caps how many
  tool-calling rounds the agent can take, which could cut off research on
  topics that genuinely need more than a few searches.

## Future Improvements

- Add a lightweight verification step that cross-checks the researched
  company's name/domain before finalizing the report.
- Replace `print()` debug statements with structured logging.
- Cache repeated search queries to reduce API usage during iterative testing.
- Extend `CompanyReport` with additional fields (industry, recent news,
  competitors) now that the validation pipeline is proven reliable.
