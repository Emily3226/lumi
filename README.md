# Lumi

Lumi is [Auxilium](https://theauxilium.ca/)'s chat assistant. One conversation covers two
jobs: pairing students with peer mentors and booking sessions with them, and tutoring
students through Waterloo math contest problems using the real contest papers.

A single chat endpoint routes each message to whichever agent should handle it, so a
student can ask for a mentor, book a slot, then switch to asking for practice problems
without ever changing pages.

---

## How it works

```
                       frontend/chat.html
                              |
                    +---------+---------+
              POST /chat          POST /contest/ask
                    |                   |
         MentorTaskAgents ---delegates--> ContestAgent
        (LangChain routing)          (intent detection)
                    |                   |
      +-------------+------+      +-----+---------------+
      |                    |      |                     |
 mentor RAG +         booking +   contest RAG      PyMuPDF crops
 GBM ranking          slots +     (vector search)  + worksheets
      |               email             |                |
      +--------------------+------------+----------------+
                           |
                     MongoDB Atlas
           (documents | vectors | GridFS PDFs)
```

Both halves share one provider-agnostic LLM layer with automatic failover, so a provider
going down or retiring a model doesn't take the app with it.

---

## Mentor matching and booking

### Finding a mentor

Matching runs as a pipeline rather than a single similarity lookup:

1. **Retrieval** (`rag/retriever.py`) embeds mentor profiles with all-MiniLM-L6-v2 and
   ranks them by cosine similarity against the mentee's request. The query is first
   expanded through a subject-alias table (`rag/subject_utils.py`), so "calc" and
   "derivatives" both reach a mentor who listed "calculus". Recall is deliberately
   widened to 3x the requested count to give the rerankers something to work with.
2. **Availability filtering** (`api/services.py`) drops mentors booked since the
   retriever's cache was built. That candidate pool is an in-process copy rebuilt only on
   restart, so this live check is what stops an already-booked mentor being offered.
3. **LLM reranking** (`rag/langchain_matcher.py`), optional. If a provider is reachable
   the candidates are reordered by the model; if not, the pipeline continues without it.
4. **Learned scoring** (`models/train.py`) rescores with a scikit-learn
   `GradientBoostingRegressor` trained on historical pairings, over four features:
   subject match, grade gap, a senior-mentor bonus, and grade similarity. Retrain any time
   via `POST /admin/train-model`, which reports R2 and MAE on a held-out 20% split.
5. **Hard filters** reject mentors too far below the mentee's grade or in the wrong
   subject, then a final sort prioritises exact subject matches and small grade gaps.

### Booking a session

Confirmation is a guided conversation, not a form. The agent walks through name, email,
slot choice, and confirmation as explicit session states, so a student can answer in their
own phrasing and the flow survives a page reload (sessions persist to disk with a TTL
sweep).

Mentors publish weekly availability as a day plus a start time, which the frontend renders
as a calendar grid. Confirmed bookings send an email through Resend, and cancelling a
booking releases both the mentor and the time slot.

### Admin

`frontend/admin.html` covers mentor CRUD, availability toggling, mentee and booking
listings, booking cancellation, slot management, and model retraining.

---

## Contest tutoring

Ask for a specific problem ("Euclid 2023 problem 5"), a topic ("geometry practice for
grade 10"), or a whole worksheet ("make me a 10-problem set from CIMC"). Lumi returns the
actual problem image cropped from the source PDF rather than a paraphrase, so diagrams
survive intact.

### Ingestion (`rag/contest_ingestor.py`)

Contest PDFs carry no machine-readable problem boundaries, so the ingestor finds them
geometrically. Every problem number in a paper is typeset in one vertical column, so it
collects candidate `N.` spans from the left margin, takes the **median x-position** of all
of them, and accepts only spans pinned to that column. The median is robust to stray
matches in a way that regex over page text is not. CIMC and CSMC additionally split into
Part A (1-6) and Part B (1-3), which get renumbered to Q7-Q9 on ingest.

### Retrieval (`rag/contest_retriever.py`, `rag/onnx_embedder.py`)

Problems are embedded and stored in MongoDB Atlas Vector Search, filterable by contest,
year, topic, and grade. Embeddings run on **ONNX Runtime** rather than
sentence-transformers: PyTorch alone costs 300-500MB resident just to import, which is
fatal on a 1GB instance. If the Atlas vector index is missing, retrieval falls back to a
streaming brute-force cosine scan that keeps only a top-k heap, so it degrades in speed
rather than dying on memory.

### Rendering (`api/contest_image_router.py`, `api/problem_set_service.py`)

Problems are cropped straight out of the source PDF with PyMuPDF and served as PNGs,
handling problems that span pages and trimming trailing whitespace. The same crops compose
into downloadable A4 worksheets with matching solution sets.

---

## Shared infrastructure

**LLM providers** (`api/llm_provider.py`) - Gemini, then Groq, then Cerebras, then
Cloudflare, skipping any without a key. All four speak the OpenAI chat-completions format,
so one request builder covers them. The chain shares a single *total* timeout budget, so
failover can't multiply a caller's worst-case wait. Reasoning models spend "thinking"
tokens that count against `max_tokens` but never reach the reply, so each provider
declares a token floor.

**Sessions and memory** (`api/session_store.py`, `api/memory_store.py`) - per-session
conversation state and extracted facts persist to disk, swept on a TTL so long-running
deployments don't accumulate them forever.

**Storage** (`api/db.py`, `rag/mongo_pdf_store.py`) - MongoDB Atlas holds mentors,
mentees, bookings, slots, and contest chunks with their vectors. Contest PDFs live in
GridFS behind a local read-through disk cache. An atomic counter helper keeps small
integer ids for bookings and slots.

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

To check that a deployment is healthy (vector index present, caching working, which path
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
| `RESEND_API_KEY` | Booking confirmation emails |
| `RESEND_FROM_EMAIL` | Sender address for those emails |
| `CONTEST_VECTOR_INDEX` | Atlas vector index name (default `contest_vector_index`) |
| `CONTEST_META_CACHE_TTL` | Corpus metadata cache, seconds (default `300`, `0` disables) |

Model IDs (`GEMINI_MODEL`, `GROQ_MODEL`, ...) can be set but are best left unset; the
defaults track current non-deprecated models. Pinning a version is how this project
previously ended up stranded on a retired model.

---

## API

**Chat**

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | Main conversational endpoint (routes between agents) |
| `GET` | `/health` | Liveness check |

**Mentors and booking**

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/match` | Ranked mentor matches |
| `POST` | `/book` | Confirm a pairing |
| `GET` | `/mentors/{name}/slots` | Available time slots |
| `GET` | `/history` | Past pairings |
| `*` | `/admin/*` | Mentor, booking, slot, and model management |

**Contest**

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/contest/ask` | Contest agent directly |
| `POST` | `/contest/search` | Raw semantic search over problems |
| `GET` | `/contest/contests` | Indexed contests and years |
| `GET` | `/contest/page-image` | Cropped PNG of a problem or solution |
| `GET` | `/contest/status` | Whether the corpus is indexed |

---

## Contests covered

| Contest | Problems/year |
| --- | --- |
| Euclid | 10 |
| CIMC, CSMC | 9 (Part A 1-6, Part B renumbered to 7-9) |
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
api/          FastAPI app: routing, agents, booking, LLM provider, PDF rendering
rag/          Ingestion, embeddings, mentor and contest retrievers
models/       Mentor-matching model training
scripts/      Ingest, migration, health-check, and index-creation tooling
frontend/     Static chat and admin UIs
tests/        Unit tests
```
