from pymongo import MongoClient
from bson import ObjectId
import unicodedata

# --- Connexion MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["labdb"]

# --- Mapping analyse → unité ---
UNITE_MAPPING = {
    "leucocytes": "/mm3",
    "plaquettes": "/mm3",
    "hematocrite": "%",
    "hemoglobine": "g/dl",
    "globules rouges": "millions/mm3",
    "vgm": "um3",
    "ccmh": "pg",
    "tcmh": "g/dl",
    "neutrophiles": "%",
    "eosinophiles": "%",
    "monocytes": "%",
    "basophiles": "%",
    "lymphocytes": "%",
}

# --- Normalisation accents ---
def enlever_accents(texte: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texte)
        if unicodedata.category(c) != "Mn"
    )

def normaliser_clef(question):
    """
    Transforme la question en clé normalisée
    ex: "Quelle est la valeur des hémoglobine ?" → "hemoglobine"
    """
    q_lower = enlever_accents(question.lower())
    for key in UNITE_MAPPING.keys():
        if key in q_lower:
            return key
    return "autres"  # valeur par défaut si non trouvée

# --- Correction valeurs ---
DIV_10 = {"hemoglobine", "hematocrite", "ccmh", "tcmh"}
DIV_100 = {"globules rouges"}
POURCENTAGE_FACTIONS = {
    "neutrophiles", "lymphocytes", "eosinophiles", "basophiles", "monocytes", "hematocrite"
}

# Intervalle normal (%) pour cellules sanguines
INTERVALLES_CELLULES = {
    "neutrophiles": (40, 90),
    "lymphocytes": (20, 45),
    "monocytes": (2, 10),
    "eosinophiles": (1, 6),
    "basophiles": (0, 1),
}

DIV_AUTO_IF_TOO_BIG = {"vgm"}  # si valeur trop grande, on divise par 10

def corriger_valeur(analyte, valeur):
    try:
        v = float(str(valeur).replace(",", "."))
    except:
        return valeur  # si pas un nombre → on laisse tel quel

    analyte_norm = analyte.lower()

    # Divisions standard
    if analyte_norm in DIV_10:
        v /= 10.0
    elif analyte_norm in DIV_100:
        v /= 100.0
    elif analyte_norm in DIV_AUTO_IF_TOO_BIG:
        if v > 100:
            v /= 10.0

    # Correction pour pourcentages si hors intervalle
    if analyte_norm in INTERVALLES_CELLULES:
        min_val, max_val = INTERVALLES_CELLULES[analyte_norm]
        # On divise par 10 tant que v est hors intervalle
        while v > max_val:
            v /= 10.0

    return v


# --- Insertion MongoDB ---
def inserer_predictions(predictions_corrigees, numero_dossier):
    """
    Insère les analyses dans une collection dynamique = numéro dossier.
    Corrige les valeurs AVANT insertion.
    """
    docs_a_inserer = []
    for pred in predictions_corrigees:
        cle = normaliser_clef(pred["question"])
        unite = UNITE_MAPPING.get(cle, "%")
        valeur_corrigee = corriger_valeur(cle, pred["answer_corrected"])  # ✅ correction ici

        docs_a_inserer.append({
            "analyse": cle,
            "value": valeur_corrigee,
            "unite": unite
        })

    if docs_a_inserer and numero_dossier:
        collection = db[numero_dossier]  # ⚡ collection dynamique par numéro dossier
        collection.insert_many(docs_a_inserer)

    return convert_objectid(docs_a_inserer)

# --- Conversion ObjectId ---
def convert_objectid(doc):
    """Convertit récursivement les ObjectId en str"""
    if isinstance(doc, list):
        return [convert_objectid(d) for d in doc]
    if isinstance(doc, dict):
        return {k: convert_objectid(v) for k, v in doc.items()}
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc
