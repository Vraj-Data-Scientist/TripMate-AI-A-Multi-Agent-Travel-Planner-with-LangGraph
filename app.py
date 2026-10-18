# Used to handle file/folder paths in a clean, cross-platform way.
from pathlib import Path

# Used to print full error details when something goes wrong.
import traceback

# ASGI server used to run the FastAPI application.(Asynchronous Server Gateway Interface.)
# FastAPI = your restaurant
# Uvicorn = waiter who receives customer orders
# ASGI = rules for how the waiter communicates with the restaurant
# parallel execution
import uvicorn

# FastAPI → Main framework class
# Request → Used to access request data in templates
from fastapi import FastAPI, Request

# HTMLResponse → Return HTML pages
# JSONResponse → Return JSON data
from fastapi.responses import HTMLResponse, JSONResponse

# Used to serve static files (CSS, JS, images).
from fastapi.staticfiles import StaticFiles

# Used to render HTML templates with Jinja2.
from fastapi.templating import Jinja2Templates

# Used to define and validate request body structure.
from pydantic import BaseModel

from backend import run_travel_agent


# Gets the folder path where this file (app.py) is located.
# Useful for finding static and templates folders.
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0"
)

# Makes the static folder available at /static URL.
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

# Sets up Jinja2 to load HTML files from the templates folder.
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# Defines the expected JSON body for the /api/travel endpoint:
class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


# Home page route.
# When user opens http://127.0.0.1:8000/, it renders index.html.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# Main API endpoint.
# Receives travel requests from the frontend.
@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        # Gets the user message and removes extra spaces.
        user_message = request_data.message.strip()

        # If message is empty, return a 400 error.
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        # Calls your LangGraph multi-agent system.
        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        # Returns the final result as JSON to the frontend.
        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "itinerary": result["itinerary"],
                "llm_calls": result["llm_calls"],
            }
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# Simple health check endpoint (useful for monitoring).
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running"
    }

# Prevents 404 error for favicon requests.
@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )