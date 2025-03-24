# src/match_score.py
from sentence_transformers import SentenceTransformer, util # type: ignore

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_fit_score(resume_text, job_description):
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(float(score[0]), 2)
