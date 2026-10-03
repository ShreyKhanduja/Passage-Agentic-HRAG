# Passage

Passage a Production level, fully local AI-powered document search and retrieval application. Upload documents, automatically structure and enrich them with LLM-generated metadata, index them using embeddings from LM Studio in a local Qdrant database, and retrieve relevant information through semantic search or an agentic retrieval workflow.

Passage combines **LLM-Enriched Metadata**, **Hierarchical Chunking**, **Agentic Retrieval**, and **HNSW vector indexing** into a completely local Retrieval-Augmented Generation (RAG) pipeline. No cloud services, external vector databases, or API keys are required.

## Features

- Upload PDF, DOCX, and TXT documents.
- **Hierarchical Chunking** that preserves document structure by identifying titles, sections, subsections, and semantically related content.
- **LLM-Enriched Metadata Generation** including summaries, keywords, titles, section hierarchy, and contextual descriptions for every chunk.
- **Agentic Retrieval** powered by LangGraph, capable of searching indexed passages, inspecting document outlines, and performing multi-step retrieval before answering.
- **Semantic Retrieval** using embeddings enhanced with structured metadata.
- **HNSW (Hierarchical Navigable Small World) Indexing** through Qdrant for fast approximate nearest-neighbor search.
- Fully local deployment using LM Studio and Qdrant Embedded.
- Single-file HTML, CSS, and JavaScript frontend with a FastAPI backend.
- Source-aware responses with citations.

---

## Architecture

```text
Documents
    ↓
Hierarchical Chunking
    ↓
LLM-Enriched Metadata Extraction
    ↓
Embedding Generation
    ↓
Qdrant HNSW Index
    ↓
Agentic Retrieval
    ↓
Context Assembly
    ↓
Local LLM Response Generation
```

---

## LLM-Enriched Metadata & Hierarchical Chunking

During ingestion, Passage uses a local LM Studio chat model to analyze document structure and generate semantically meaningful chunks.

Instead of splitting documents into arbitrary text segments, the LLM identifies:

- Document titles
- Sections
- Subsections
- Semantic boundaries
- Important concepts and themes

For every chunk, the model generates:

- Title
- Section
- Subsection
- Summary
- Keywords
- Contextual metadata

These metadata fields are embedded alongside the chunk content, creating richer retrieval signals than vector-only search.

If semantic structuring is unavailable or fails, Passage falls back to overlapping chunks of:

- 300 words per chunk
- 50 word overlap

By default:

- Embeddings use `Qwen/Qwen3-Embedding-0.6B-GGUF` via LM Studio (1024 dimensions)
- Semantic chunking uses `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF`

---

## Agentic Retrieval

Passage includes a LangGraph-powered **Agentic Retrieval** workflow.

Unlike traditional RAG systems that perform a single vector search, the retrieval agent can dynamically decide how to gather context before generating an answer.

The agent can:

- Search relevant passages
- Inspect document outlines
- Identify relevant documents
- Perform multiple retrieval steps
- Refine searches based on intermediate findings
- Filter retrieval to specific source files
- Generate cited responses

Available retrieval tools include:

- `search_book_passages`
- `get_document_outline`

This approach improves answer quality for larger document collections and complex questions.

---

## HNSW Indexing

Passage uses Qdrant's **HNSW (Hierarchical Navigable Small World)** vector index for efficient approximate nearest-neighbor search.

During ingestion:

1. Documents are hierarchically chunked.
2. Chunks are enriched with LLM-generated metadata.
3. Embeddings are generated using LM Studio.
4. Vectors are stored in a local Qdrant collection.
5. Qdrant automatically builds an HNSW graph.

During retrieval:

1. The query is embedded using the same embedding model.
2. Qdrant traverses the HNSW graph.
3. The most relevant chunks are returned without comparing against every stored vector.

Results include:

- Retrieved passage
- Relevance score
- Source filename
- Chunk metadata

HNSW configuration and optimization are handled automatically by Qdrant. This project uses Qdrant's default graph configuration without custom tuning.

The local vector database is stored in:

```text
backend/qdrant_data
```

---

## Fully Local Execution

Passage runs entirely on local infrastructure.

The following components execute locally:

- Document ingestion
- Hierarchical chunking
- Metadata enrichment
- Embedding generation
- HNSW indexing
- Vector retrieval
- Agentic retrieval workflows
- Response generation

No external services are required.

Benefits include:

- Complete data privacy
- Offline operation
- No recurring API costs
- Low-latency retrieval
- Full control over models and data storage

---

## Requirements

- Python 3.10 or newer
- [LM Studio](https://lmstudio.ai/)
- `Qwen/Qwen3-Embedding-0.6B-GGUF`
- `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF`
- Local LM Studio server enabled (default port `1234`)
- No Qdrant server required
- No cloud credentials required

Default model configuration:

| Component | Model |
|------------|--------|
| Embeddings | `Qwen/Qwen3-Embedding-0.6B-GGUF` |
| Semantic Chunking | `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF` |
| Metadata Enrichment | `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF` |
| Agentic Retrieval | `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF` |

A separate chat model can optionally be configured using `AGENT_MODEL`.

---

## Setup

Create and activate a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Create the environment file:

```powershell
Copy-Item backend\.env.example backend\.env
```

Update model identifiers as necessary to match the names displayed in LM Studio.

Download and load:

- `Qwen/Qwen3-Embedding-0.6B-GGUF`
- `lmstudio-community/Qwen3-4B-Instruct-2507-GGUF`

Start the LM Studio local server.

No API keys are required.

---

## Run

Start the backend:

```powershell
cd backend
uvicorn main:app --reload
```

Start the frontend:

```powershell
cd frontend
python -m http.server 5173
```

Open:

```text
http://localhost:5173
```

The frontend expects the API at:

```text
http://localhost:8000
```

This can be changed from the application sidebar.

---

## How To Use

1. Open **Add Documents**.
2. Upload PDF, DOCX, or TXT files.
3. Wait for indexing to complete.
4. Search using natural language queries.
5. Review retrieved passages and source citations.

### Agent Mode

Use the **Ask Agent** tab to interact with the collection through the LangGraph retrieval agent.

The agent:

- Retrieves passages from Qdrant
- Inspects document structure when needed
- Uses retrieval tools dynamically
- Produces grounded responses with citations

Conversation history remains in the browser session and is not persisted by the backend.

---

## Why Passage?

Passage combines four advanced retrieval technologies into a lightweight local-first platform:

- **LLM-Enriched Metadata**
- **Hierarchical Chunking**
- **Agentic Retrieval**
- **HNSW Vector Indexing**

The result is a fully local RAG system that delivers significantly more accurate retrieval and grounded responses than traditional fixed-chunk vector search while maintaining complete ownership of data, models, and infrastructure.

---

## Project Structure

```text
backend/       FastAPI API and document retrieval service
frontend/      Single-file web interface
```

---

## License

MIT
