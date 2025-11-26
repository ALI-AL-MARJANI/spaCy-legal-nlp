import sys
from pathlib import Path
import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin


def load_data(path: Path):
    "Load a .spacy file into a list of Doc objects"
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs


def evaluate_model(model_path: str, dev_path: str):
    "Evaluate a trained spaCy pipeline against the dev set"
    model_path = Path(model_path)
    dev_path = Path(dev_path)

    nlp = spacy.load(model_path)

    dev_docs = load_data(dev_path)

    scorer = Scorer()
    for doc in dev_docs:
        pred = nlp(doc.text)
        scorer.score(pred, doc)

    results = scorer.score
    print("Evaluation Results:")
    print("Text Classification F-score:", results.get("textcat_f", "N/A"))
    print("NER F-score:", results.get("ents_f", "N/A"))
    print("Precision:", results.get("ents_p", "N/A"))
    print("Recall:", results.get("ents_r", "N/A"))

    print("\nSample Predictions:")
    for doc in dev_docs[:3]:
        pred = nlp(doc.text)
        print("\nText:", doc.text[:150], "...")
        print("Predicted clauses:", pred.cats)
        print("Named entities:", [(e.text, e.label_) for e in pred.ents])
        print("Risk score:", pred._.risk_score)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/evaluate.py model-best dev.spacy")
        sys.exit(1)

    model_dir = sys.argv[1]
    dev_file = sys.argv[2]

    evaluate_model(model_dir, dev_file)
