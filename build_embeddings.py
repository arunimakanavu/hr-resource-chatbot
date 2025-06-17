import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load JSON employee data
with open("employees.json", "r", encoding="utf-8") as f:
    data = json.load(f)

employees = data["employees"]

# Prepare texts for embedding (you can customize this representation)
def employee_to_text(emp):
    return f"{emp['name']} has skills in {', '.join(emp['skills'])}, " \
           f"{emp['experience_years']} years experience, " \
           f"worked on {', '.join(emp['projects'])}, " \
           f"currently {emp['availability']}."

texts = [employee_to_text(emp) for emp in employees]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
print("Creating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# Convert to float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, "employee_index.faiss")

with open("employee_metadata.pkl", "wb") as f:
    pickle.dump(employees, f)

print("FAISS index and metadata saved.")
