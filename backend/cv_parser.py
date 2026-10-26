import io
import os
import re
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pypdf

load_dotenv()

# Structured schema requested for Gemini parsing
class ParsedResume(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    skills: List[str] = Field(description="List of key technical and domain skills")
    experience_summary: str = Field(description="Brief 2-3 sentence overview of candidate's career experience")
    chunks: List[str] = Field(
        description="Granular 300-500 character resume chunks covering education, study gaps, background transitions, projects, and accomplishments"
    )


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text from PDF bytes using pypdf."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    extracted_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text.append(page_text)
    return "\n".join(extracted_text).strip()


def chunk_text_heuristically(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Generates 300-500 character overlapping chunks as fallback/supplement."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks = []
    start = 0
    text_len = len(cleaned)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - overlap
    return chunks


def parse_resume(file_bytes: bytes) -> ParsedResume:
    """
    Extracts text from PDF bytes and uses Gemini with structured Pydantic schema
    to return a ParsedResume object, with model fallback resilience.
    """
    raw_text = extract_text_from_pdf(file_bytes)
    if not raw_text:
        return ParsedResume(
            candidate_name="Applicant",
            skills=[],
            experience_summary="No text could be extracted from the document.",
            chunks=[]
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        chunks = chunk_text_heuristically(raw_text)
        return ParsedResume(
            candidate_name="Candidate",
            skills=["Software Engineering", "Problem Solving"],
            experience_summary=raw_text[:250],
            chunks=chunks
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (
            "Analyze the candidate resume below. Extract the candidate's full name, list of technical "
            "skills, a concise 2-3 sentence experience summary, and slice the content into granular "
            "300-500 character resume chunks. Crucially, make sure chunks cover:\n"
            "1. Educational degrees, institutions, and graduation years.\n"
            "2. Any study gaps, career breaks, or transitions between study and career.\n"
            "3. Key personal projects, hobbies, and technical accomplishments.\n\n"
            f"RESUME TEXT:\n{raw_text[:8000]}"
        )

        candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash"]
        last_error = None
        response = None

        for model_id in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ParsedResume,
                        temperature=0.1,
                    ),
                )
                if response and response.text:
                    break
            except Exception as ex:
                last_error = ex
                continue

        if not response or not response.text:
            raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

        # Parse structured json
        parsed = ParsedResume.model_validate_json(response.text)
        
        # Ensure chunks are non-empty and well-formed
        if not parsed.chunks or len(parsed.chunks) < 2:
            fallback_chunks = chunk_text_heuristically(raw_text)
            parsed.chunks = fallback_chunks if fallback_chunks else [raw_text[:400]]

        return parsed
    except Exception as e:
        print(f"[cv_parser] Gemini parsing error: {e}")
        chunks = chunk_text_heuristically(raw_text)
        first_line = raw_text.splitlines()[0][:50].strip() if raw_text.splitlines() else "Candidate"
        return ParsedResume(
            candidate_name=first_line if len(first_line) > 2 else "Candidate",
            skills=["Software Development", "Problem Solving"],
            experience_summary=raw_text[:300],
            chunks=chunks if chunks else [raw_text[:400]]
        )
