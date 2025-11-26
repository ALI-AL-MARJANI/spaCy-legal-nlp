from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI(title="Legal NLP Assistant")

# Load the trained spaCy model when the server starts
# (we will change this path after training)
MODEL_PATH = "../training/model-best"
nlp = spacy.load(MODEL_PATH)


class ContractRequest(BaseModel):
    text: str


@app.post("/analyze_contract")
def analyze_contract(data: ContractRequest):
    "Analyze a contract and return clauses, entities, and risk score"
    doc = nlp(data.text)

    # Extract clause predictions (textcat)
    clause_scores = {
        label: float(score)
        for label, score in doc.cats.items()
    }

    # Extract entities
    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]

    # Extract risk score from custom component
    risk_score = float(doc._.risk_score)

    return {
        "clauses": clause_scores,
        "entities": entities,
        "risk_score": risk_score,
    }


@app.get("/")
def root():
    return {"message": "Legal NLP Assistant API is running "}
