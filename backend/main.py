import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.vector_engine import ResumeStore
from backend.cv_parser import parse_resume
from backend.stt_service import transcribe_audio_file
from backend.interview_engine import (
    generate_next_question,
    update_rolling_summary,
    generate_interview_evaluation_report,
)

# Load credentials from .env
load_dotenv()
load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="Dynamic AI Interview System", version="1.0.0")

# Enable CORS for cross-origin or local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session manager
sessions: Dict[str, Dict[str, Any]] = {}


class GenerateJDRequest(BaseModel):
    role: str


class StartInterviewRequest(BaseModel):
    session_id: str


@app.post("/api/generate-jd")
async def generate_jd_endpoint(payload: GenerateJDRequest):
    """
    Generates a modern 3-part structured Job Description based on target role with model fallback.
    """
    role = payload.role.strip()
    if not role:
        raise HTTPException(status_code=400, detail="Target role is required.")

    api_key = os.getenv("GEMINI_API_KEY")
    prompt = f"""Generate a comprehensive, structured 3-part Job Description for the professional role: '{role}'.
Cleanly format the text using exactly these three distinct markdown sections:
### 1. Role Overview & Objectives
(Provide 2-3 sentences outlining the role's mission and business context)

### 2. Core Responsibilities & Impact
(Provide 4-5 bullet points covering day-to-day work, engineering, and delivery)

### 3. Required Technical Skills & Qualifications
(Provide 4-5 bullet points covering languages, technologies, and soft skills)
"""
    candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"]
    jd_text = None

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            for model_id in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                    )
                    if response and response.text:
                        jd_text = response.text.strip()
                        break
                except Exception:
                    continue
        except Exception as e:
            print(f"[main] Gemini client error: {e}")

    if not jd_text:
        jd_text = f"""### 1. Role Overview & Objectives
As a {role}, you will contribute to building modern software solutions and collaborative products with quality and reliability.

### 2. Core Responsibilities & Impact
- Develop and maintain software components, APIs, and user-facing features.
- Participate actively in team code reviews, knowledge sharing, and agile planning.
- Solve everyday technical bugs and assist in ongoing system enhancements.
- Learn and adopt modern frameworks and engineering best practices.

### 3. Required Technical Skills & Qualifications
- Solid foundational understanding of programming languages and web concepts.
- Eagerness to learn new libraries and tools quickly.
- Clear communication, curious mindset, and strong collaborative team attitude."""

    return {"job_description": jd_text}


@app.post("/api/upload-resume")
async def upload_resume_endpoint(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    job_role: Optional[str] = Form(None),
):
    """
    Parses resume PDF, builds in-memory FAISS vector index, calculates match score.
    Returns candidate metadata and gatekeeper eligibility (match_score >= 10.0%).
    """
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await resume.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Parse resume text and extract structured entities
    parsed_cv = parse_resume(file_bytes)

    # 2. Build in-memory FAISS vector index from resume chunks
    resume_store = ResumeStore(parsed_cv.chunks)

    # 3. Calculate semantic match score (0.0 to 100.0%)
    match_score = resume_store.compute_match_score(job_description)
    
    # Gatekeeper 10% threshold rule
    eligible = match_score >= 10.0

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "job_description": job_description,
        "job_role": job_role or "Software Developer",
        "resume_store": resume_store,
        "candidate_name": parsed_cv.candidate_name,
        "skills": parsed_cv.skills,
        "experience_summary": parsed_cv.experience_summary,
        "match_score": match_score,
        "eligible": eligible,
        "history": [],
        "summary": "",
        "turn_count": 0,
        "report": None,
        "completed": False,
    }

    return {
        "session_id": session_id,
        "candidate_name": parsed_cv.candidate_name,
        "skills": parsed_cv.skills,
        "match_score": match_score,
        "eligible": eligible,
    }


@app.post("/api/interview/start")
async def start_interview_endpoint(payload: StartInterviewRequest):
    """
    Initializes the interview session and generates an easy, welcoming opening question.
    """
    session = sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not session.get("eligible", False):
        raise HTTPException(
            status_code=403,
            detail="Your profile does not sufficiently match the core requirements for this position at this time."
        )

    resume_store: ResumeStore = session["resume_store"]
    top_chunks = resume_store.retrieve_relevant_chunks(session["job_description"], k=3)

    first_question = generate_next_question(
        job_description=session["job_description"],
        resume_chunks=top_chunks,
        conversation_history=[],
        candidate_latest_answer=None,
        rolling_summary="",
        candidate_name=session.get("candidate_name"),
        job_role=session.get("job_role"),
    )

    session["turn_count"] = 1
    session["history"] = [{
        "turn": 1,
        "question": first_question,
        "answer": None,
    }]

    return {
        "session_id": payload.session_id,
        "question": first_question,
        "turn_count": 1,
        "total_turns": 5,
        "interview_completed": False,
    }


@app.post("/api/interview/submit-answer")
async def submit_answer_endpoint(
    session_id: str = Form(...),
    text_answer: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """
    Accepts candidate response via audio transcription or direct text.
    Maintains 5-turn limit: on turn 5 completion, generates comprehensive scoring report.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Process answer from audio or text
    candidate_transcript = ""
    if audio_file:
        audio_bytes = await audio_file.read()
        candidate_transcript = transcribe_audio_file(audio_bytes, filename_hint=audio_file.filename or "audio.webm")
    
    if not candidate_transcript and text_answer:
        candidate_transcript = text_answer.strip()

    if not candidate_transcript:
        raise HTTPException(status_code=400, detail="No verbal or text response received.")

    # Record answer in current turn
    current_turn_index = len(session["history"]) - 1
    if current_turn_index >= 0:
        session["history"][current_turn_index]["answer"] = candidate_transcript
        last_question = session["history"][current_turn_index]["question"]
    else:
        last_question = "Initial inquiry."

    # Update rolling conversation summary
    session["summary"] = update_rolling_summary(
        previous_summary=session["summary"],
        question=last_question,
        answer=candidate_transcript,
    )

    # CHECK 5-TURN LIMIT: If candidate just answered question 5, conclude the interview!
    if len(session["history"]) >= 5:
        session["completed"] = True
        report = generate_interview_evaluation_report(
            candidate_name=session.get("candidate_name", "Applicant"),
            job_role=session.get("job_role", "Software Engineer"),
            job_description=session.get("job_description", ""),
            conversation_history=session["history"],
            rolling_summary=session.get("summary", ""),
        )
        session["report"] = report

        return {
            "candidate_transcript": candidate_transcript,
            "turn_count": 5,
            "total_turns": 5,
            "interview_completed": True,
            "report": report,
        }

    # Otherwise, generate the next question (turns 2, 3, 4, 5)
    resume_store: ResumeStore = session["resume_store"]
    relevant_chunks = resume_store.retrieve_relevant_chunks(candidate_transcript, k=3)

    turn_num = len(session["history"]) + 1
    if turn_num == 2 and resume_store.chunks:
        edu_chunks = resume_store.retrieve_relevant_chunks("education university study degree background transition", k=2)
        relevant_chunks = list(dict.fromkeys(relevant_chunks + edu_chunks))

    next_question = generate_next_question(
        job_description=session["job_description"],
        resume_chunks=relevant_chunks,
        conversation_history=session["history"],
        candidate_latest_answer=candidate_transcript,
        rolling_summary=session["summary"],
        candidate_name=session.get("candidate_name"),
        job_role=session.get("job_role"),
    )

    session["turn_count"] = turn_num
    session["history"].append({
        "turn": session["turn_count"],
        "question": next_question,
        "answer": None,
    })

    return {
        "candidate_transcript": candidate_transcript,
        "next_question": next_question,
        "turn_count": session["turn_count"],
        "total_turns": 5,
        "interview_completed": False,
    }


@app.get("/api/interview/report")
async def get_report_endpoint(session_id: str):
    """Returns the evaluation report for a completed session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    report = session.get("report")
    if not report:
        raise HTTPException(status_code=400, detail="Evaluation report not available.")
    return report


# Mount static frontend directory at the root
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
