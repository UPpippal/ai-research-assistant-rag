from tavily import TavilyClient
from dotenv import load_dotenv
import os

# load environment variables
load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query):

    results = client.search(query=query, max_results=3)

    context = ""
    sources = []

    for r in results["results"]:
        context += r["content"] + "\n"
        sources.append(r["url"])

    return context, sources