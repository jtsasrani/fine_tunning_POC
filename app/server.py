import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.inference import DWPInferenceEngine
import uvicorn

app = FastAPI(title="DWP CMG RAG Inference API")

# Lazy initialize engine
engine = None

class QueryRequest(BaseModel):
    query: str

class ContextResponse(BaseModel):
    paragraph_id: str
    source_doc: str
    text: str
    score: float

class ConfigResult(BaseModel):
    response: str
    time: float

class QueryResponse(BaseModel):
    query: str
    retrieved_contexts: List[ContextResponse]
    retrieval_time: float
    config_a: ConfigResult
    config_b: ConfigResult
    config_c: ConfigResult

@app.on_event("startup")
def startup_event():
    global engine
    print("Preloading RAG models on GPU...")
    engine = DWPInferenceEngine()
    print("RAG models successfully loaded!")

@app.post("/api/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    global engine
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        t0 = time.time()
        retrieved = engine.retrieve_context(request.query)
        retrieval_time = time.time() - t0
        
        retrieved_data = []
        for r in retrieved:
            retrieved_data.append(ContextResponse(
                paragraph_id=r["chunk"]["paragraph_id"],
                source_doc=r["chunk"]["source_doc"],
                text=r["chunk"]["text"],
                score=round(r["score"], 4)
            ))
            
        res_a, time_a = engine.generate("mistral_tuned", request.query, retrieved)
        res_b, time_b = engine.generate("qwen_tuned", request.query, retrieved)
        res_c, time_c = engine.generate("mistral_base", request.query, retrieved)
        
        return QueryResponse(
            query=request.query,
            retrieved_contexts=retrieved_data,
            retrieval_time=round(retrieval_time, 3),
            config_a=ConfigResult(response=res_a, time=round(time_a, 2)),
            config_b=ConfigResult(response=res_b, time=round(time_b, 2)),
            config_c=ConfigResult(response=res_c, time=round(time_c, 2))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
