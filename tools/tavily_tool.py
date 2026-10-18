from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key= os.getenv("TAVILY_API_KEY")
)

def tavily_search(query):
    response = client.search(
        query= query,
        max_results= 3
    )

    results = []

    # client.search() returns a dictionary. Inside it, keys are : [query, result, response_time, request_id]
    # the key "results" contains a list of dictionaries, where each dictionary represents one search result.

    # [
    #     {"title": "...", "url": "...", "content": "..."},
    # ]

    # The second argument 1 makes the counting start from 1 instead of the default 0.

    for i, r in enumerate(response["results"], 1):
        title   = r.get("title", "Unknown")
        url     = r.get("url", "")
        snippet = r.get("content", "").strip()
        # Keep only the first 300 characters to avoid wall-of-text
        # The .rsplit(" ", 1)[0] :: split from right side and by space and only once (coz , 1)
        if len(snippet) > 200:
            snippet = snippet[:200].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)





