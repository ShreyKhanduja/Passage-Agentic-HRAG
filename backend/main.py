from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from agent_service import RetrievalAgent
from rag_service import RAGService
import uvicorn

app = FastAPI(title="HNSW RAG API")

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()
retrieval_agent = RetrievalAgent(rag_service)

class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5

class SearchResult(BaseModel):
    score: float
    content: str
    filename: str
    title: str = ""
    section: str = ""
    subsection: Optional[str] = None
    header_path: List[str] = Field(default_factory=list)
    summary: str = ""
    keywords: List[str] = Field(default_factory=list)
    chunk_type: str = ""
    chunk_index: int = 0

class IngestResponse(BaseModel):
    message: str
    chunks_added: int

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=12)

class AgentChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "HNSW RAG System"}

@app.post("/chat", response_model=AgentChatResponse)
def chat_with_agent(request: AgentChatRequest):
    try:
        result = retrieval_agent.chat(
            request.message,
            [turn.model_dump() for turn in request.history],
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    try:
        content_bytes = await file.read()
        filename_lower = file.filename.lower()
        
        # Handle different file types
        if filename_lower.endswith('.pdf'):
            # Extract text from PDF
            from PyPDF2 import PdfReader
            import io
            pdf_reader = PdfReader(io.BytesIO(content_bytes))
            content = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    content += page_text + "\n\n"
            print(f"Extracted {len(content)} characters from PDF ({len(pdf_reader.pages)} pages)")
            
        elif filename_lower.endswith('.docx'):
            # Extract text from Word document
            from docx import Document
            import io
            doc = Document(io.BytesIO(content_bytes))
            content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            print(f"Extracted {len(content)} characters from DOCX")
            
        else:
            # Try to decode as text file
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    content = content_bytes.decode(encoding)
                    print(f"Successfully decoded file with {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise HTTPException(status_code=400, detail="Could not decode file. Supported formats: .txt, .pdf, .docx")
        
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")
            
        chunks = rag_service.ingest_document(content, file.filename)
        return {"message": f"Successfully ingested {file.filename}", "chunks_added": chunks}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
def search_documents(query: SearchQuery):
    try:
        results = rag_service.search(query.query, query.limit)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
