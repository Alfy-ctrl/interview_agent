import io
import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv
from fpdf import FPDF
from starlette.testclient import TestClient

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure environment variables are loaded
load_dotenv()
load_dotenv(root_dir / "backend" / ".env")

from backend.main import app
from backend.vector_engine import ResumeStore
from backend.interview_engine import generate_next_question


def create_synthetic_pdf(
    candidate_name: str = "Alex Mercer",
    skills: str = "Python, FastAPI, PostgreSQL, Docker, Redis, Distributed Systems",
    experience: str = "Senior Backend Engineer with 5 years of experience architecting high-throughput REST APIs and microservices. Scaled transaction processing pipelines and optimized SQL queries."
) -> bytes:
    """Helper to generate an in-memory synthetic PDF resume using fpdf2."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(text=candidate_name, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text=f"Core Competencies: {skills}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(w=0, h=6, text=f"Professional Summary:\n{experience}")
    
    pdf_output = pdf.output()
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1")
    elif isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    return pdf_output


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ====================================================================
# Test 1: Verify JD generation endpoint returns valid structured requirements
# ====================================================================
def test_jd_generation_endpoint(client):
    payload = {"role": "Python Backend Engineer"}
    response = client.post("/api/generate-jd", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert "job_description" in data
    jd = data["job_description"]
    assert len(jd) > 80
    # Verify structured sections exist
    assert "Overview" in jd or "Responsibilities" in jd or "Skills" in jd or "Qualifications" in jd


# ====================================================================
# Test 2: Test FAISS vector matcher (>= 10% threshold & foreign chunks)
# ====================================================================
def test_faiss_vector_matcher():
    technical_chunks = [
        "Architected scalable backend microservices with Python, FastAPI, and PostgreSQL handling 10,000 requests per second.",
        "Implemented Redis distributed caching layer and optimized slow SQL query execution times by 45%.",
        "Configured Docker containers and Kubernetes clusters for CI/CD continuous deployment pipelines."
    ]
    store = ResumeStore(technical_chunks)
    
    python_jd = """
    We are seeking a Python Backend Engineer to build scalable microservices using FastAPI, 
    manage PostgreSQL databases, and deploy with Docker and Kubernetes.
    """
    
    match_score = store.compute_match_score(python_jd)
    # Technical match should easily clear the 10.0% threshold
    assert match_score >= 10.0, f"Expected match score >= 10.0%, got {match_score}%"

    # Test retrieval
    relevant = store.retrieve_relevant_chunks("PostgreSQL database caching", k=2)
    assert len(relevant) > 0
    assert any("PostgreSQL" in c or "Redis" in c for c in relevant)

    # Empty store should return 0.0%
    empty_store = ResumeStore([])
    assert empty_store.compute_match_score(python_jd) == 0.0

    # Foreign / completely unrelated text with non-technical profile
    unrelated_chunks = ["Experienced pastry chef baking sourdough bread and French patisserie in artisan bakeries."]
    unrelated_store = ResumeStore(unrelated_chunks)
    unrelated_score = unrelated_store.compute_match_score(python_jd)
    assert unrelated_score < 40.0, f"Unrelated score should be low, got {unrelated_score}"


# ====================================================================
# Test 3: Synthetic PDF creation + upload test verifying PDF text parsing
# ====================================================================
def test_resume_upload_and_parsing(client):
    pdf_bytes = create_synthetic_pdf()
    files = {"resume": ("alex_mercer_resume.pdf", pdf_bytes, "application/pdf")}
    data = {
        "job_description": "Looking for a Senior Python Developer with experience in FastAPI, Docker, and distributed systems."
    }
    
    response = client.post("/api/upload-resume", data=data, files=files)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    result = response.json()
    assert "session_id" in result
    assert "candidate_name" in result
    assert "skills" in result
    assert "match_score" in result
    assert "eligible" in result
    
    # Assert gatekeeper evaluation
    assert result["match_score"] >= 10.0
    assert result["eligible"] is True
    assert len(result["candidate_name"]) > 0


# ====================================================================
# Test 4: Verify Groq question generation returns a grounded interview question
# ====================================================================
def test_groq_question_generation():
    jd = "Python Backend Developer needed to architect async APIs and distributed task queues."
    resume_chunks = [
        "Alex Mercer built high-throughput REST APIs in FastAPI and managed Celery async job queues."
    ]
    
    # Opening question
    question = generate_next_question(
        job_description=jd,
        resume_chunks=resume_chunks,
        conversation_history=[],
        candidate_latest_answer=None,
    )
    
    assert isinstance(question, str)
    assert len(question) > 15
    assert not question.startswith("Certainly")
    assert not question.startswith("Here is")

    # Follow-up question
    follow_up = generate_next_question(
        job_description=jd,
        resume_chunks=resume_chunks,
        conversation_history=[{"question": question, "answer": "I configured Celery with Redis broker and handled message retries."}],
        candidate_latest_answer="I configured Celery with Redis broker and handled message retries.",
    )
    assert isinstance(follow_up, str)
    assert len(follow_up) > 15


# ====================================================================
# Test 5: End-to-end integration test
# ====================================================================
def test_end_to_end_interview_flow(client):
    # Step 1: Generate JD
    jd_res = client.post("/api/generate-jd", json={"role": "Full Stack Developer"})
    assert jd_res.status_code == 200
    jd = jd_res.json()["job_description"]

    # Step 2: Upload Synthetic PDF
    pdf_bytes = create_synthetic_pdf(
        candidate_name="Jordan Reed",
        skills="TypeScript, React, Python, FastAPI, Tailwind CSS",
        experience="Built end-to-end web applications with modern React frontends and scalable FastAPI backends."
    )
    files = {"resume": ("jordan_resume.pdf", pdf_bytes, "application/pdf")}
    upload_res = client.post("/api/upload-resume", data={"job_description": jd}, files=files)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    
    session_id = upload_data["session_id"]
    assert upload_data["eligible"] is True
    assert upload_data["match_score"] >= 10.0

    # Step 3: Start Interview
    start_res = client.post("/api/interview/start", json={"session_id": session_id})
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert "question" in start_data
    assert start_data["turn_count"] == 1
    first_q = start_data["question"]

    # Step 4: Submit Text Answer (Turn 1 answered -> Advance to Turn 2)
    candidate_answer = "I built full-stack applications with React and Python and focused on clear coding style."
    answer_res = client.post(
        "/api/interview/submit-answer",
        data={
            "session_id": session_id,
            "text_answer": candidate_answer
        }
    )
    assert answer_res.status_code == 200
    answer_data = answer_res.json()
    assert answer_data["turn_count"] == 2
    assert "next_question" in answer_data
    assert not answer_data.get("interview_completed", False)

    # Step 5: Advance through turns 2, 3, 4, 5 to trigger evaluation report
    for turn in range(2, 5):
        resp = client.post(
            "/api/interview/submit-answer",
            data={
                "session_id": session_id,
                "text_answer": f"Answer for screening question {turn} demonstrating foundational knowledge."
            }
        )
        assert resp.status_code == 200
        data = resp.json()
        assert not data.get("interview_completed", False)

    # Submit final answer for Question 5 -> completes the 5-question interview
    final_resp = client.post(
        "/api/interview/submit-answer",
        data={
            "session_id": session_id,
            "text_answer": "When debugging, I isolate the issue using logging, write a test case, and verify the fix."
        }
    )
    assert final_resp.status_code == 200
    final_data = final_resp.json()
    assert final_data.get("interview_completed") is True
    assert "report" in final_data
    
    report = final_data["report"]
    assert "overall_score" in report
    assert "parameters" in report
    assert "sentence_framing" in report["parameters"]
    assert "score" in report["parameters"]["sentence_framing"]
    assert "technical_basics" in report["parameters"]
    assert len(report.get("turn_reviews", [])) == 5
