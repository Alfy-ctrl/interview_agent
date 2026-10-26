# AI Interview Agent

An intelligent, multi-turn AI interview platform with real-time audio transcription, resume parsing with vector search (FAISS), dynamic question generation with Groq LLM, and comprehensive candidate evaluation reporting.

## Features
- **CV / Resume Parsing**: Extracts skills, work history, and granular candidate experience chunks from uploaded PDFs using Gemini and PyPDF.
- **Vector Search Engine**: Indexes candidate experience chunks using `SentenceTransformer` and FAISS for semantic grounding.
- **Dynamic Interview Engine**: Generates adaptive, multi-turn technical and behavioral interview questions tailored to the resume and job description.
- **Speech-to-Text (STT)**: Transcribes candidate voice responses in real time using `faster-whisper`.
- **Candidate Evaluation**: Generates detailed scorecards and assessment summaries upon interview completion.
- **Modern Interactive Web Interface**: Single-page frontend with voice capture and live interview feedback.

## Architecture & Directory Structure
```
├── backend/
│   ├── .env.example            # Environment variables template
│   ├── main.py                 # FastAPI application & REST endpoints
│   ├── cv_parser.py            # PDF extraction and structured parsing
│   ├── interview_engine.py     # Adaptive question generation & scoring
│   ├── vector_engine.py        # FAISS embedding index for candidate background
│   └── stt_service.py          # Faster-Whisper audio transcription
├── frontend/
│   └── index.html              # Responsive web dashboard
├── tests/
│   └── test_interview_system.py # End-to-end integration tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Alfy-ctrl/interview_agent.git
cd interview_agent
```

### 2. Configure Environment Variables
Create a `.env` file in the `backend/` directory:
```bash
cp backend/.env.example backend/.env
```
Fill in your API keys in `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install Python Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Run the Application
Start the backend server:
```bash
uvicorn backend.main:app --reload --port 8000
```
Open `frontend/index.html` in your web browser or access the FastAPI service at `http://localhost:8000`.

## Testing
Run the test suite:
```bash
pytest tests/ -v
```
