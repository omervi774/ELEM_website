import stanza
import re
# Download Hebrew model (only first time)
stanza.download("he")

# Load pipeline with NER
nlp = stanza.Pipeline("he", processors="tokenize,ner")

# Map Stanza entity types to Hebrew labels
type_map = {
    "PER": "שם",
    "ORG": "ארגון",
    "LOC": "מקום",
    "GPE": "מדינה/עיר",
    "FAC": "כתובת"
}
def anonymize_text(text):
    phone_pattern = re.compile(r"05\d[- ]?\d{7}")
    id_pattern = re.compile(r"\b\d{9}\b")
    #replace phone number
    text = phone_pattern.sub("<טלפון>", text)
    # Replace ID
    text = id_pattern.sub("<תעודת זהות>", text)

    doc = nlp(text)

    # Dictionary to store assigned IDs for each unique PERSON
    person_map = {}
    person_counter = 1

    anonymized_text = text

    for ent in doc.ents:
        hebrew_label = type_map.get(ent.type, ent.type)

        if ent.type == "PER":  # Person counter
            if ent.text not in person_map:
                # Assign a new ID if this person hasn't appeared before
                person_map[ent.text] = f"<{hebrew_label}{person_counter}>"
                person_counter += 1
            replacement = person_map[ent.text]
        elif ent.type == "ID":
            replacement = "<תעודת זהות>"
        else:
            replacement = f"<{hebrew_label}>"

        # Replace in text
        anonymized_text = anonymized_text.replace(ent.text, replacement)

    return anonymized_text, person_map