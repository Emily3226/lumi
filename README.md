# Lumi

Lumi is [Auxilium](https://theauxilium.ca/)'s chat assistant. It does two things behind
one conversation: matches students with peer mentors and books sessions with them, and
tutors students through Waterloo math contest problems using the real contest PDFs.

The contest side is a RAG pipeline over **1,743 problems** spanning 11 Waterloo contests.
Ask for a specific problem ("Euclid 2023 problem 5"), a topic ("geometry practice for
grade 10"), or a whole worksheet ("make me a 10-problem set from CIMC") and it pulls the
actual problem image out of the source PDF rather than paraphrasing it.

---

## How it works

```
                       frontend/chat.html
                              │
                    ┌─────────┴─────────┐
              POST /chat          POST /contest/ask
                    │                   │
         MentorTaskAgents ──delegates──► ContestAgent
        (LangChain routing)          (intent detection)
                    │                   │
      ┌─────────────┤                   ├──────────────┬────────────────┐
      ▼             ▼                   ▼              ▼                ▼
 mentor RAG   scikit-learn        contest RAG    PyMuPDF crops   problem-set PDFs
 (retriever)  ranking model    (vector search)   (page images)   (worksheets)
      │             │                   │              │                │
      └─────────────┴─────────┬─────────┴──────────────┴────────────────┘
                              ▼
                        MongoDB Atlas
              (documents · vectors · GridFS PDFs)
```

Both agents share a provider-agnostic LLM layer with automatic failover, so a single
provider going down or retiring a model doesn't take the app with it.

### The pieces

**Contest ingestion** (`rag/contest_ingestor.py`)

Contest PDFs have no machine-readable problem boundaries, so the ingestor finds them
geometrically. Every problem number in a given paper is typeset in one vertical column,
so it collects candidate `N.` spans from the left margin, takes the **median x-position**
of all of them, and accepts only spans pinned to that column. The median is robust to
stray matches in a way that regex over page text is not. CIMC and CSMC additionally split
into Part A (1–6) and Part B (1–3), which get renumbered to Q7–Q9 on ingest.

**Retrieval** (`rag/contest_retriever.py`, `rag/onnx_embedder.py`)

Problems are embedded with all-MiniLM-L6-v2 and stored in MongoDB Atlas Vector Search.
Embeddings run on **ONNX Runtime** rather than sentence-transformers: PyTorch alone costs
300–500MB resident just to import, which is fatal on a 1GB instance. Queries filter by
contest, year, topic, and grade. If the Atlas vector index is missing, retrieval falls
back to a streaming brute-force cosine scan that keeps only a top-*k* heap, so it degrades
in speed rather than crashing on memory.

**Problem rendering** (`api/contest_image_router.py`, `api/problem_set_service.py`)

Rather than reproducing problem text (and losing diagrams), Lumi crops the actual region
of the source PDF with PyMuPDF and serves it as a PNG, handling problems that span pages
and trimming trailing whitespace. The same crops compose into downloadable A4 worksheets
with matching solution sets.

**Mentor matching** (`rag/retriever.py`, `models/train.py`, `api/services.py`)

Mentor profiles are embedded and retrieved by semantic similarity, then rescored by a
scikit-learn `GradientBoostingRegressor` trained on historical pairings, factoring in
subject alignment, grade gap, and qualifications. Availability is filtered live at request
time so a mentor booked since the cache was built can't be offered again.

**LLM providers** (`api/llm_provider.py`)

Gemini → Groq → Cerebras → Cloudflare, in that order, skipping any without a key. All
four speak the OpenAI chat-completions format, so one request builder covers them. The
chain shares a single *total* timeout budget, so failover can't multiply a caller's
worst-case wait. Reasoning models spend "thinking" tokens that count against `max_tokens`
but never appear in the reply, so each provider declares a token floor.

---

## Running it

Requires Python 3.10+ (Pydantic models use `int | None` at runtime) and a MongoDB Atlas
cluster.

```bash
pip install -r requirements.txt
```

Create a `.env` in the repo root:

```bash
MONGODB_URI=mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/
GEMINI_API_KEY=your_key_from_ai_studio
```

Then:

```bash
python scripts/warm_embedding_cache.py     # download the ONNX model once (~90MB)
python scripts/ingest_contests.py          # parse contest PDFs into MongoDB
python scripts/create_vector_index.py      # create the Atlas Vector Search index
uvicorn api.main:app --reload
```

Open <http://localhost:8000>; the API redirects to the chat UI.

To check a deployment is healthy (vector index present, caching working, which path
serves a query):

```bash
python scripts/check_deploy_health.py
```

### Configuration

| Variable | Purpose |
| --- | --- |
| `MONGODB_URI` | Atlas connection string (**required**) |
| `MONGODB_DB_NAME` | Database name (default `lumi`) |
| `GEMINI_API_KEY` | Primary LLM provider (free tier, no card) |
| `GROQ_API_KEY` | Fallback provider |
| `LLM_PROVIDER` | Pin a single provider, e.g. `gemini` |
| `LLM_PROVIDER_ORDER` | Override the failover order |
| `CONTEST_VECTOR_INDEX` | Atlas vector index name (default `contest_vector_index`) |
| `CONTEST_META_CACHE_TTL` | Corpus metadata cache, seconds (default `300`, `0` disables) |
| `RESEND_API_KEY` | Booking confirmation emails |

Model IDs (`GEMINI_MODEL`, `GROQ_MODEL`, …) can be set but are best left unset; the
defaults track current non-deprecated models. Pinning a version is how this project
previously ended up stranded on a retired model.

---

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | Main conversational endpoint (routes between agents) |
| `POST` | `/contest/ask` | Contest agent directly |
| `POST` | `/contest/search` | Raw semantic search over problems |
| `GET` | `/contest/contests` | Indexed contests and years |
| `GET` | `/contest/page-image` | Cropped PNG of a problem or solution |
| `GET` | `/contest/status` | Whether the corpus is indexed |
| `POST` | `/match` | Ranked mentor matches |
| `POST` | `/book` | Confirm a pairing |
| `GET` | `/mentors/{name}/slots` | Available time slots |
| `*` | `/admin/*` | Mentor, booking, and slot management |
| `GET` | `/health` | Liveness check |

---

## Contests covered

| Contest | Problems/year |
| --- | --- |
| Euclid | 10 |
| CIMC, CSMC | 9 (Part A 1–6, Part B renumbered to 7–9) |
| Fryer, Galois, Hypatia | 4 |
| Gauss 7/8, Pascal, Cayley, Fermat | 25 |

Problems are auto-tagged into algebra, number theory, geometry, combinatorics, sequences,
inequalities, trigonometry, and logic.

---

## Tests

```bash
python -m unittest discover -s tests
```

---

## Layout

```
api/          FastAPI app: routing, agents, LLM provider, PDF rendering
rag/          Ingestion, embeddings, retrievers
models/       Mentor-matching model training
scripts/      Ingest, migration, health-check, and index-creation tooling
frontend/     Static chat and admin UIs
tests/        Unit tests
```
