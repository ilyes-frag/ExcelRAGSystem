from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.rag_pipeline import query_rag_system,initialize_rag_pipeline_once
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html for any unknown routes (acts like a fallback for React Router)

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
class QueryRequest(BaseModel):
    query: str

## Define request body model
class QueryRequest(BaseModel):
    query: str



# initialize RAG system once during app startup
@app.on_event("startup")
async def startup_event():
    initialize_rag_pipeline_once()

# Define the query endpoint
@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        answer = query_rag_system(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the query: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    pass