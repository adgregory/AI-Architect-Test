import spacy


nlp = spacy.load("en_core_web_sm")


def extract_names(text: str) -> list[str]:
    """Extract person names from text using spaCy NER."""
    doc = nlp(text)

    names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text)

    return names


def extract_names_with_positions(text: str) -> list[dict]:
    """Extract person names with their character positions in text."""
    doc = nlp(text)

    results = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            results.append({
                "name": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_,
            })

    return results
