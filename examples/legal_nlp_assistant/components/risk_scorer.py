from spacy.language import Language
from spacy.tokens import Doc


@Language.factory("risk_scorer")
def create_risk_scorer(nlp, name, model=None):
    "Factory that returns the RiskScorer component"
    return RiskScorer(model)


class RiskScorer:
    """
    A customized spaCy component that assigns a simple risk score to a document
    The score is not meant to be legally accurate but serves as an example of how to build a custom component

    The component uses:
    - predicted clauses or texts (textcat)
    - extracted legal entities (ner)
    - simple heuristic patterns
    """

    def __init__(self, model=None):
        self.model = model

    def __call__(self, doc: Doc):
        "Adds a `_.risk_score` attribute to the doc, containing a float between 0 and 1 indicating the risk level"
        score = 0.0

        # higher risk if TERMINATION clause is missing from textcat predictions
        if "TERMINATION" not in doc.cats:
            score += 0.2

        # check for "unlimited liability" keywords...
        text_lower = doc.text.lower()
        risky_patterns = [
            "unlimited liability",
            "full liability",
            "sole responsibility",
            "without limitation",
        ]
        if any(pattern in text_lower for pattern in risky_patterns):
            score += 0.4

        # large payment amounts increase risk
        for ent in doc.ents:
            if ent.label_ == "PAYMENT_AMOUNT":
                if any(c in ent.text for c in ["$", "€"]):
                    # heuristic: if amount > 100000 we increase risk
                    digits = "".join([d for d in ent.text if d.isdigit()])
                    if digits.isdigit() and int(digits) > 100000:
                        score += 0.3
        score = min(1.0, score)
        doc._.risk_score = score
        return doc


# Register the doc extension attribute (only once)
if not Doc.has_extension("risk_score"):
    Doc.set_extension("risk_score", default=0.0)
