# Prescription DDI Checker

A privacy-first polypharmacy **Drug-Drug Interaction (DDI) Checker** designed specifically for Indian prescriptions.

---

## 📌 Problem Statement

In India, doctors predominantly prescribe medications using commercial brand names (e.g., *Dolo 650*, *Telmikind-AM*, *Augmentin*, *Pan-D*) rather than generic chemical salt names. When patients consult multiple specialists or take concurrent medications (polypharmacy), two major risks arise:
1. **Adverse Drug-Drug Interactions (DDI):** Two or more prescribed drugs interact negatively, reducing efficacy or causing severe adverse clinical effects.
2. **Duplicate Active Salt Overdose:** Different branded medications often contain identical generic compounds, leading to accidental double-dosing and toxicity.

Most clinical DDI databases index drugs only by generic chemical names (INN/USAN). This project bridges that gap by normalizing Indian brand names to active generic salts and checking for drug interactions locally and privately.

---

## 🚀 Key Features

- **Prescription OCR & Extraction:** Extracts medicine names directly from prescription images using EasyOCR with confidence scoring.
- **Brand-to-Salt Normalization:** Maps Indian commercial brand names into their underlying active chemical salts via exact database matching and PostgreSQL trigram fuzzy matching (`pg_trgm`).
- **Duplicate Salt Overdose Alerts:** Detects when multiple prescribed brands contain overlapping active ingredients.
- **Hybrid RAG Interaction Detection:** Evaluates drug pairs using a two-tier hybrid search in Supabase (`pgvector` cosine similarity with `all-MiniLM-L6-v2` embeddings + exact metadata filtering).
- **Clinical Severity Classification:** Flags interactions into distinct risk tiers:
  - 🔴 **Critical:** Contraindicated or life-threatening.
  - 🟠 **Major:** Substantial risk requiring close clinical monitoring.
  - 🟡 **Moderate:** Mild risk requiring caution.
  - 🟢 **Safe:** No known adverse interactions found.
- **Privacy-Preserving Report Synthesis:** Generates human-readable, patient-friendly safety reports using a local Ollama LLM (`phi3.5`) or automated fallback templates without transmitting health data to third-party cloud APIs.

---

## ⚙️ How It Works

```
┌─────────────────────────┐
│   Prescription Image    │
└────────────┬────────────┘
             │
             ▼
   [ 1. OCR Extraction ]          ── EasyOCR detects medicine text with confidence scores
             │
             ▼
   [ 2. Brand Normalization ]     ── Resolves brand names to active chemical salts
             │                       (Catalog lookup + pg_trgm fuzzy matching)
             ▼
   [ 3. Safety Analysis ]         ── Scans for duplicate salts across different brands
             │
             ▼
   [ 4. Hybrid RAG Checker ]      ── Queries Supabase pgvector store for pairwise interactions
             │                       (Exact metadata match + 384-d all-MiniLM-L6-v2 embeddings)
             ▼
   [ 5. Report Synthesis ]        ── Local LLM (Ollama phi3.5) / Template Engine produces
             │                       structured safety report
             ▼
┌─────────────────────────┐
│ Final Safety Report Card│
└─────────────────────────┘
```

---

## 📁 Repository Structure

```
prescription_ddi_checker/
├── README.md
├── backend/
│   ├── .env.example                         # Template environment variables
│   ├── requirements.txt                     # Python dependencies
│   ├── app/
│   │   ├── main.py                          # FastAPI application & model lifespan loader
│   │   ├── config.py                        # Pydantic settings & environment configuration
│   │   ├── db/
│   │   │   └── supabase_client.py           # Supabase connection client
│   │   ├── models/
│   │   │   └── responses.py                 # Pydantic API response schemas
│   │   ├── routers/                         # API route controllers
│   │   └── services/                        # Business logic (OCR, normalizer, RAG checker)
│   ├── data/
│   │   ├── sample_indian_medicines.csv      # Indian commercial brands to generic salts
│   │   └── sample_ddi.csv                   # Curated drug-drug interaction records
│   └── supabase/
│       └── migrations/
│           ├── 001_create_medicines_catalog.sql  # Brands table & trigram index
│           ├── 002_create_interactions_rag.sql   # pgvector DDI interaction store
│           └── 003_create_match_rpc.sql          # Hybrid search RPC database function
```

---

## 🛠️ Setup & Installation

### Prerequisites
- **Python:** 3.11+
- **Supabase:** Free-tier or local project (requires `pgvector` and `pg_trgm` extensions)
- **Ollama (Optional):** For local LLM clinical explanations (`phi3.5`)

### 1. Database Setup (Supabase)
Run the SQL migrations in order in your Supabase SQL Editor:
1. `backend/supabase/migrations/001_create_medicines_catalog.sql`
2. `backend/supabase/migrations/002_create_interactions_rag.sql`
3. `backend/supabase/migrations/003_create_match_rpc.sql`

### 2. Environment Configuration
Create a `.env` file inside the `backend/` directory from `.env.example`:

```bash
cd backend
cp .env.example .env
```

Configure the following variables in `backend/.env`:
```ini
# Supabase Project Credentials
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-role-key

# Ollama (Local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3.5

# OCR & Embedding Configurations
OCR_CONFIDENCE_THRESHOLD=75.0
OCR_GPU=false
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.82
DEBUG=false
```

### 3. Virtual Environment & Dependencies
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Ollama Setup (Optional)
If running local LLM synthesis:
```bash
ollama pull phi3.5
ollama run phi3.5
```
*(If Ollama is offline, the backend automatically uses deterministic template fallbacks).*

---

## 🏃 Running the Application

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

- **API Base URL:** `http://127.0.0.1:8000`
- **Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`
