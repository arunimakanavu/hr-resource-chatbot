from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Load embedding model & FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("employee_index.faiss")
with open("employee_metadata.pkl", "rb") as f:
    employee_data = pickle.load(f)

app = FastAPI(title="HR Resource Query Chatbot")


# --- Models ---
class ChatQuery(BaseModel):
    query: str
    top_k: int = 3


# --- Helper functions ---
def embed_query(query):
    return model.encode([query]).astype("float32")

def search_employees(query, top_k=3):
    emb = embed_query(query)
    distances, indices = index.search(emb, top_k)

    results = []
    for i in indices[0]:
        emp = employee_data[i]
        results.append(emp)
    return results

def format_profiles(employees):
    return "\n\n".join([
        f"{e['name']} - Skills: {', '.join(e['skills'])}, "
        f"{e['experience_years']} years experience, "
        f"Projects: {', '.join(e['projects'])}, "
        f"Available: {e['availability'].capitalize()}"
        for e in employees
    ])

def generate_response(user_query, employees):
    context = format_profiles(employees)
    prompt = f"""You are an HR assistant. The user asked: "{user_query}".
Based on the employee data below, suggest suitable candidates with reasoning.

Employee data:
{context}

Response:"""

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama2",
        "prompt": prompt,
        "stream": False
    })

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="LLM generation failed")

    return response.json()["response"].strip()


# --- API Endpoints ---
@app.post("/chat")
def chat_with_bot(payload: ChatQuery):
    employees = search_employees(payload.query, payload.top_k)
    if not employees:
        return {"response": "No suitable candidates found."}
    reply = generate_response(payload.query, employees)
    return {"response": reply}


@app.get("/employees/search")
def employee_search(query: str = Query(...), top_k: int = 3):
    employees = search_employees(query, top_k)
    return {"results": employees}
