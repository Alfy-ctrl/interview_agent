import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model


class ResumeStore:
    def __init__(self, chunks: list[str]):
        self.chunks = [c.strip() for c in chunks if c and c.strip()]
        self.model = get_embed_model()
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        self._build_index()

    def _build_index(self):
        if not self.chunks:
            return
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))

    def compute_match_score(self, job_description: str) -> float:
        """
        Computes overall similarity score (0.0 to 100.0%) between the resume
        and the provided job description.
        """
        if not self.chunks or self.index.ntotal == 0 or not job_description or not job_description.strip():
            return 0.0

        jd_embedding = self.model.encode([job_description], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(jd_embedding)
        
        k = min(5, self.index.ntotal)
        distances, indices = self.index.search(jd_embedding.astype(np.float32), k)
        
        valid_scores = [float(s) for s in distances[0] if s > 0]
        if not valid_scores:
            return 0.0
            
        # Top-k match average converted to percentage
        avg_score = float(np.mean(valid_scores))
        score_pct = round(avg_score * 100.0, 1)
        return min(100.0, max(0.0, score_pct))

    def retrieve_relevant_chunks(self, topic_query: str, k: int = 3) -> list[str]:
        """
        Retrieves top k grounded chunks for question generation.
        """
        if not self.chunks or self.index.ntotal == 0 or not topic_query or not topic_query.strip():
            return self.chunks[:k]

        query_embedding = self.model.encode([topic_query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        top_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results
