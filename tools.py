import json
import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()


# --------------------------------
# LOAD CASE DATA
# --------------------------------

with open("data/cases.json", "r") as file:
    cases = json.load(file)


# --------------------------------
# TOOL 1 - CURRENT CASE
# --------------------------------

def get_case(case_id):
    for case in cases:
        if case["case_id"] == case_id:
            return case

    return None


# --------------------------------
# TOOL 2 - CASE HISTORY
# --------------------------------

def get_case_history(case_id):
    with open("data/history.json", "r") as file:
        history = json.load(file)

    case_history = []

    for event in history:
        if event["case_id"] == case_id:
            case_history.append(event)

    return case_history


# --------------------------------
# TOOL 3 - KNOWLEDGE / RAG
# --------------------------------

def search_knowledge(question):
    with open("data/knowledge/prior_authorization.txt", "r") as file:
        document = file.read()

    document_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=document
    ).data[0].embedding

    question_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name="pharmacy_knowledge"
    )

    collection.upsert(
        ids=["pa_policy"],
        documents=[document],
        embeddings=[document_embedding]
    )

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=1
    )

    return results["documents"][0][0]