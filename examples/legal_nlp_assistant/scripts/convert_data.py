import json
import random
import sys
from pathlib import Path
import spacy
from spacy.tokens import DocBin


def load_json(path: Path):
    "Load raw JSON data from disk as a list of examples"
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_docs(nlp, examples):
    """Convert raw examples into spaCy Doc objects."""
    docs = []
    for eg in examples:
        text = eg["text"]
        cats = eg.get("cats", {})
        entities = eg.get("entities", [])
        
        doc = nlp.make_doc(text)

        # Text classification labels (clause types)
        doc.cats = cats

        # Entity spans
        spans = []
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]
            span = doc.char_span(start, end, label=label)
            if span is not None:
                spans.append(span)

        doc.ents = spans
        docs.append(doc)
    return docs


def save_docbin(docs, path: Path):
    "Serialize a list of Doc objects to disk as a .spacy file"
    doc_bin = DocBin(store_user_data=True)
    for doc in docs:
        doc_bin.add(doc)
    doc_bin.to_disk(path)


def main(raw_path: str, train_path: str, dev_path: str, split_ratio: float = 0.8):
    raw_path = Path(raw_path)
    train_path = Path(train_path)
    dev_path = Path(dev_path)

    nlp = spacy.blank("en")

    data = load_json(raw_path)
    random.shuffle(data)

    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    dev_data = data[split_idx:]

    train_docs = create_docs(nlp, train_data)
    dev_docs = create_docs(nlp, dev_data)

    save_docbin(train_docs, train_path)
    save_docbin(dev_docs, dev_path)

    print(f"Saved {len(train_docs)} training docs to {train_path}")
    print(f"Saved {len(dev_docs)} dev docs to {dev_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/convert_data.py raw_data.json train.spacy dev.spacy")
        sys.exit(1)

    raw_data_file = sys.argv[1]
    train_file = sys.argv[2]
    dev_file = sys.argv[3]

    main(raw_data_file, train_file, dev_file)
