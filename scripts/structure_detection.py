import spacy
from spacy.matcher import PhraseMatcher, Matcher
import json
import os

# Chargement du modèle
nlp = spacy.load("fr_core_news_lg")


def get_headers_config(nlp):
    """
    Définit les listes de mots-clés basées sur ton fichier BSV 2022.
    """
    # 1. Les Cultures (Sections principales)
    crops_list = ["Colza", "Tournesol", "Soja", "Betterave", "Orge", "Blé", "Maïs", "Pois"]

    # 2. Les Bio-agresseurs et thématiques (Sous-sections)
    # Basé sur ton fichier : "Pièges à limaces" , "Pucerons verts" [cite: 283]
    topics_list = [
        "Stades", "Pièges", "Limaces", "Altise", "Pucerons", "Charançon",
        "Méligèthes", "Sclérotinia", "Phoma", "Adventices", "Oiseaux"
    ]

    # 3. Le marqueur spécifique d'analyse de risque
    risk_markers = ["Analyse du risque", "Niveau de risque"]

    # Création des patterns pour PhraseMatcher (correspondance exacte de séquence)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # Insensible à la casse

    # On ajoute les patterns
    matcher.add("HEADER_CROP", [nlp.make_doc(text) for text in crops_list])
    matcher.add("HEADER_TOPIC", [nlp.make_doc(text) for text in topics_list])
    matcher.add("HEADER_RISK", [nlp.make_doc(text) for text in risk_markers])

    return matcher


def extract_structure(text, matcher):
    doc = nlp(text)
    matches = matcher(doc)

    # Structure de données pour stocker le résultat
    structured_data = {}

    # Variables d'état pour savoir où on se trouve dans la hiérarchie
    current_crop = "CONTEXTE_GENERAL"
    current_topic = "General"
    last_pos = 0

    # On trie les matchs par position
    matches.sort(key=lambda x: x[1])

    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start:end]

        # Vérification contextuelle : Est-ce vraiment un titre ?
        # Règle heuristique : Un titre est souvent court et suivi d'un saut de ligne ou ":"
        # On regarde la ligne contenant le match
        sent = span.sent
        is_likely_header = len(sent.text.split()) < 15 or ":" in sent.text

        if not is_likely_header:
            continue  # On ignore si le mot apparaît au milieu d'une longue phrase

        # --- Sauvegarde du contenu précédent ---
        # On prend tout le texte entre la fin du dernier titre et le début du nouveau
        content = doc[last_pos:start].text.strip()

        if content:
            if current_crop not in structured_data:
                structured_data[current_crop] = {}
            if current_topic not in structured_data[current_crop]:
                structured_data[current_crop][current_topic] = []

            structured_data[current_crop][current_topic].append(content)

        # --- Mise à jour de l'état (Changement de section) ---
        header_text = span.text.upper()  # Normalisation

        if rule_id == "HEADER_CROP":
            current_crop = header_text
            current_topic = "General"  # Reset du topic quand on change de culture

        elif rule_id == "HEADER_TOPIC":
            # Si on trouve un ravageur (ex: Limace), cela devient le sous-topic
            current_topic = header_text

        elif rule_id == "HEADER_RISK":
            # Cas spécial : on veut garder le texte de l'analyse de risque séparément ou tagué
            current_topic = "ANALYSE_RISQUE"

        last_pos = end  # Le prochain contenu commencera après ce titre

    # --- Récupération du dernier bloc de texte ---
    final_content = doc[last_pos:].text.strip()
    if final_content:
        if current_crop not in structured_data:
            structured_data[current_crop] = {}
        if current_topic not in structured_data[current_crop]:
            structured_data[current_crop][current_topic] = []
        structured_data[current_crop][current_topic].append(final_content)

    return structured_data


# --- Exécution sur ton fichier ---
def run_extraction():
    # Chemin vers ton fichier texte nettoyé
    # Note: Assure-toi que le chemin correspond à ton environnement
    input_file = "data/processed/bfc/cleanned/2022/bsv_gc_n_1_du_23_08_22.txt"

    # Lecture (Simulation avec le contenu que tu as envoyé)
    # Dans ton vrai code : with open(input_file, 'r', encoding='utf-8') as f: text = f.read()
    # Ici j'utilise une partie du texte que tu as fourni pour l'exemple
    text_sample = """
    Colza :
    Mise en place des pièges dans les colzas.
    Peu de ravageurs pour le moment.

    Tournesol :
    Ne pas récolter trop tôt.

    Colza
    Altise des crucifères ou petites altises
    Il s'agit d'un petit coléoptère noir.
    - Analyse du risque :
    Pour les colzas qui viennent d'être semés, le risque est faible.

    Limaces
    Le colza est particulièrement appétant.
    """

    matcher = get_headers_config(nlp)
    data = extract_structure(text_sample, matcher)

    print(json.dumps(data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    run_extraction()