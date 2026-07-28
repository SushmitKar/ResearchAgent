**Research Agent**

What is an agent?

An agent is the combination of an LLM and the ability to call outside tools and a loop that lets it decide what to do next based on the results.

Unlike an LLM, where the user simply asks a question and the model answers back
from the learning it did during training.

So the flow becomes : 
1. LLM reads your query: "Research XYZ"
2. LLM decides: "I need to search the web first"
3. LLM calls a search tool (this is real code running, not the LLM itself)
4. Search results come back as text
5. LLM reads results, decides: "I should also check their website"
6. LLM calls a website-fetch tool
7. LLM now has real information, writes a structured summary

This **request->decide->act->observe->decide-again** pattern
is called agent loop. The agent can continue this loop until it has enough information to provide a comprehensive answer
to the user's query.


**APIs -- decided to use**
1. LLM: Anthropic API
2. Search: Tavily

What is a "tool," technically?

A tool is two things paired together:
1. A real Python function that does something (e.g., calls Tavily's search API and returns text).
2. A JSON schema describing that function — its name, what it does, what parameters it takes — written in a format the LLM can read.

The LLM never runs Python code. Its a text model.
1. You give the LLM a list of tool schemas ("here's what you're allowed to call")
2. LLM decides: "I want to call search_web with query='LumiQ company'"
3. LLM outputs THIS DECISION as structured JSON (not code, not execution)
4. Your Python code reads that JSON, sees "oh it wants search_web", 
   actually calls the real search_web() function
5. Your code takes the function's return value, sends it back to the LLM as a new message
6. LLM continues, now with real search results in its context


**_Why two prompts instead of your one: your original prompt asked the model to do two jobs in the same breath — "decide whether to call a tool" and "eventually output strict JSON" — while tools were still attached to that same call. That's a lot of competing instructions for the model to juggle at once, and it's the direct cause of the plain-English replies and malformed JSON you kept seeing._**