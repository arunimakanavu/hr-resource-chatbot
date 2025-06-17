import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and employee metadata
index = faiss.read_index("employee_index.faiss")
with open("employee_metadata.pkl", "rb") as f:
    employee_data = pickle.load(f)

# Function to format the output nicely
def format_employee(emp):
    return (
        f"**{emp['name']}**\n"
        f"• Skills: {', '.join(emp['skills'])}\n"
        f"• Experience: {emp['experience_years']} years\n"
        f"• Projects: {', '.join(emp['projects'])}\n"
        f"• Availability: {emp['availability'].capitalize()}\n"
    )

# Query function
def query_employees(user_query, top_k=3):
    query_embedding = model.encode([user_query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        emp = employee_data[idx]
        results.append(format_employee(emp))

    return "\n---\n".join(results)


# Example usage
if __name__ == "__main__":
    print("HR Chatbot: Enter your query (e.g. 'Find Python devs with 3+ years experience'):\n")
    user_input = input(">> ")
    print("\n Top Candidates:\n")
    print(query_employees(user_input, top_k=3))
