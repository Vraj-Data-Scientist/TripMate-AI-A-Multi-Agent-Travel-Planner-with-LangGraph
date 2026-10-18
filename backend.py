import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

# These lines fix SSL Certificate problems that sometimes occur(path issue)
# when making HTTPS requests (like calling AviationStack API).
# Returns the path of a trusted certificate file provided by the certifi package
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# TypedDict is used to define the structure of the state that will be shared
# between all agents/nodes in LangGraph.
from typing import TypedDict, Annotated

# Used with Annotated to combine values (like adding messages)(we dont want to replace it with previous)
import operator

# Generate unique IDs (usually for thread/session IDs)
import uuid

# Connect to PostgreSQL database
import psycopg
from psycopg.rows import dict_row

# Build the multi-agent graph (nodes + edges)
from langgraph.graph import StateGraph, START, END

# Save conversation state/memory in PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver

# Create and handle different types of chat messages
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# Use Groq LLM (fast models like Llama, Mixtral, etc.)
from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )
    # if it is not in url we add it becoz it's remote server
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY,
    max_tokens=800,
    temperature=0
)


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }




# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Keep the itinerary concise.
Include only the most useful information.
Do not repeat raw flight or hotel details.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be concise.
- Do not invent flight prices.
- Mention that AviationStack provides live/status data, not ticket prices.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

# Creates a connection to the PostgreSQL database. and gives connection
# Automatically save changes (no need to call commit())
# Return database rows as dictionaries instead of tuples
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

# Creates a checkpointer using the database connection.
# This allows LangGraph to:
#
# Save the state after every step
# Remember previous conversation
# Resume from where it left off using thread_id

checkpointer = PostgresSaver(_conn)

# Creates the required tables in PostgreSQL (if they don’t already exist).
# LangGraph needs some tables to store:
#
# Conversation state
# Checkpoints
# Thread information

checkpointer.setup()

# Compiles the graph with memory support.
# Each thread_id can continue the previous conversation
travel_graph = graph.compile(checkpointer=checkpointer)



# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):

    # If no thread_id is provided, create a new unique one.
    # This is used to identify and remember the conversation.
    # So the frontend can continue the same conversation later.
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    # Creates a configuration dictionary for LangGraph.
    # The thread_id tells the checkpointer which conversation memory to use.
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
