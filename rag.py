import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Read our knowledge document
with open("data/knowledge/prior_authorization.txt", "r") as file:
    document = file.read()

# Create embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=document
)

embedding = response.data[0].embedding

# Create Chroma database
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="pharmacy_knowledge"
)

# Store document + embedding
collection.add(
    ids=["pa_policy"],
    documents=[document],
    embeddings=[embedding]
)

print("Document stored in ChromaDB successfully.")

# User question
question = "What should we do if insurance asks for more medical information?"

# Convert question into an embedding
question_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

question_embedding = question_response.data[0].embedding

# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=1
)

print()
print("Question:", question)
print()
print("Retrieved document:")
print(results["documents"][0][0])

context = results["documents"][0][0]

rag_response = client.responses.create(
    model="gpt-5.6-luna",
    input=f"""
Answer the question using only the provided policy.

Policy:
{context}

Question:
{question}
"""
)

print()
print("RAG Answer:")
print(rag_response.output_text)