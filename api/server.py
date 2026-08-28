# api/server.py

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from api.pipeline import RAGPipeline

# One pipeline per tenant — loaded at startup
PIPELINES: dict[str, RAGPipeline] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create default pipeline at startup
    PIPELINES["default"] = RAGPipeline(tenant_id="default")
    print("RAGPipeline ready.")
    yield
    # Cleanup
    for p in PIPELINES.values():
        p.embedder.close()

app = FastAPI(title="KnowledgeOS API", lifespan=lifespan)


# --- Request/Response models ---

class QueryRequest(BaseModel):
    question:           str
    tenant_id:          str   = "default"
    check_faithfulness: bool  = False   # set True for regulated domains


class IngestRequest(BaseModel):
    path:      str
    tenant_id: str = "default"


# --- Endpoints ---

@app.get("/health")
def health():
    return {
        "status":   "ok",
        "tenants":  list(PIPELINES.keys()),
        "chunks":   {t: p.stats["n_chunks"] for t, p in PIPELINES.items()},
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    if req.tenant_id not in PIPELINES:
        PIPELINES[req.tenant_id] = RAGPipeline(tenant_id=req.tenant_id)

    pipeline = PIPELINES[req.tenant_id]
    result   = pipeline.ingest(req.path)
    return result


@app.post("/query")
def query(req: QueryRequest):
    if req.tenant_id not in PIPELINES:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant '{req.tenant_id}' not found. POST /ingest first."
        )
    pipeline = PIPELINES[req.tenant_id]
    result   = pipeline.query(
        req.question,
        check_faithfulness=req.check_faithfulness,
    )
    return result


@app.get("/stats/{tenant_id}")
def stats(tenant_id: str):
    if tenant_id not in PIPELINES:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return PIPELINES[tenant_id].stats