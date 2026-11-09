# Prescription DDI Checker

Privacy-first polypharmacy Drug-Drug Interaction checker for Indian prescriptions.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Ollama (for local LLM)
- Supabase account (free tier works)

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Linux/WSL2
pip install -r requirements.txt
cp .env.example .env            # Edit with your Supabase credentials
uvicorn app.main:app --reload
```

### Ollama Setup
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3.5
export OLLAMA_KEEP_ALIVE=5m
```

### Supabase Setup
Run the SQL migrations in order in the Supabase SQL Editor:
1. `backend/supabase/migrations/001_create_medicines_catalog.sql`
2. `backend/supabase/migrations/002_create_interactions_rag.sql`
3. `backend/supabase/migrations/003_create_match_rpc.sql`

