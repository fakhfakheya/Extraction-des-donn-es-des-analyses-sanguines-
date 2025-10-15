# api_ocr.py
import cv2
import pytesseract
import unicodedata
import re
from difflib import get_close_matches

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Champs et unités
champs_cibles = [
    "leucocytes", "hématies", "plaquettes", "hémoglobine", "hématocrite",
    "neutrophiles", "lymphocytes", "eosinophiles", "basophiles", "monocytes",
    "globules rouges", "globules blancs", "vgm", "ccmh", "tcmh"
]

unites_valides = {
    "g/dL", "pg", "µm³", "um³", "fL", "%", "10^3/uL", "10^6/uL",
    "mm³", "10*3/uL", "10*6/uL", "/mm³", "10*3/mm³", "10*6/mm³", "/ml", "g/dl",
    "Pg", "10*6/µl", "/µl", "/ul", "millions/mm", "M/mm³", "femtolites", "10⁶/mm³",
    "10p6/mm³", "10p3/mm³", "10 p 6/mm³", "10 p 3/mm³"
}

champs_sans_accents = [
    re.sub(r'[^a-z0-9\s]', '', unicodedata.normalize('NFD', c).lower())
    for c in champs_cibles
]

questions = {c: f"Quelle est la valeur des {c} ?" if c not in {"vgm","ccmh","tcmh"}
             else f"Quelle est la valeur du {c} ?" for c in champs_cibles}

# ---------------- FONCTIONS UTILES ----------------
def normaliser(mot):
    mot = unicodedata.normalize('NFD', mot)
    return ''.join(c for c in mot if unicodedata.category(c) != 'Mn').lower()

def nettoyer_et_enlever_accents(texte):
    texte = unicodedata.normalize('NFD', texte.lower())
    return re.sub(r'[^a-z0-9\s]', '', ''.join(c for c in texte if unicodedata.category(c) != 'Mn'))

def est_unite_complexe(mot):
    pattern = r'^\d+p\d+\/mm[3³]$'
    pattern_general = r'^\d+[a-z]+\d+\/[a-z0-9/³*^]+$'
    return bool(re.match(pattern, mot.lower()) or re.match(pattern_general, mot.lower()))

def separer_chiffres_unites(texte):
    mots = texte.split()
    mots_corriges = []
    for mot in mots:
        if est_unite_complexe(mot):
            mots_corriges.append(mot)
        else:
            mot_sep = re.sub(r'(\d)([a-zA-Zµ/%])', r'\1 \2', mot)
            mot_sep = re.sub(r'([a-zA-Zµ/%])(\d)', r'\1 \2', mot_sep)
            mots_corriges.append(mot_sep)
    return ' '.join(mots_corriges)

def corriger_mot_ocr(mot):
    mot = mot.replace("'", "").replace("`", "")
    mot_clean = nettoyer_et_enlever_accents(mot)
    match = get_close_matches(mot_clean, champs_sans_accents, n=1, cutoff=0.7)
    return champs_cibles[champs_sans_accents.index(match[0])] if match else mot

def corriger_unite_intelligente(mot):
    mot_normalise = normaliser(mot)
    correspondances = get_close_matches(mot_normalise, [normaliser(u) for u in unites_valides], n=1, cutoff=0.6)
    if correspondances:
        index = [normaliser(u) for u in unites_valides].index(correspondances[0])
        return list(unites_valides)[index]
    return mot

def corriger_ligne_intelligente(ligne):
    ligne = separer_chiffres_unites(ligne)
    mots = ligne.split()
    mots_corriges = []
    for mot in mots:
        mot_corrige = corriger_mot_ocr(mot)
        if mot_corrige == mot:
            mot_corrige = corriger_unite_intelligente(mot)
        mots_corriges.append(mot_corrige)
    return ' '.join(mots_corriges)

def extraire_valeurs_selon_champ(champ, ligne):
    ligne_sep = separer_chiffres_unites(ligne)
    valeurs = re.findall(r'\d+[.,]?\d*', ligne_sep)
    valeurs = [v.replace(',', '.') for v in valeurs]
    if not valeurs:
        return None
    return valeurs[0]

def fusionner_chiffres(tokens):
    nouvelle_liste = []
    i = 0
    while i < len(tokens):
        if i+1 < len(tokens) and tokens[i].replace(',', '').isdigit() and tokens[i+1].replace(',', '').isdigit():
            fusion = tokens[i] + tokens[i+1]
            nouvelle_liste.append(fusion)
            i += 2
        else:
            nouvelle_liste.append(tokens[i])
            i += 1
    return nouvelle_liste

# ------------------- TRAITEMENT IMAGE -------------------
def traiter_image(image_path: str, filename: str):
    image = cv2.imread(image_path)
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='fra')
    n_boxes = len(data['text'])

    lines, tolerance_y = {}, 15
    for i in range(n_boxes):
        mot = data['text'][i].strip()
        conf = int(data['conf'][i])
        if mot == '' or conf < 0:
            continue
        y, x = data['top'][i], data['left'][i]
        found = False
        for key in lines:
            if abs(key - y) <= tolerance_y:
                lines[key].append({
                    'word': mot,
                    'x': x,
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i]
                })
                found = True
                break
        if not found:
            lines[y] = [{
                'word': mot,
                'x': x,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            }]

    lignes_corrigees, lignes_mots_pos = [], []
    for y in sorted(lines):
        mots_tries = sorted(lines[y], key=lambda m: m['x'])
        ligne = " ".join(m['word'] for m in mots_tries)
        ligne_corrigee = corriger_ligne_intelligente(ligne)
        lignes_corrigees.append(ligne_corrigee)
        lignes_mots_pos.append(mots_tries)

    qas = []
    for champ, question in questions.items():
        for idx, ligne in enumerate(lignes_corrigees):
            if nettoyer_et_enlever_accents(champ) in nettoyer_et_enlever_accents(ligne):
                valeur = extraire_valeurs_selon_champ(champ, ligne)
                if valeur:
                    mots_pos = lignes_mots_pos[idx]
                    tokens = [m['word'] for m in mots_pos]
                    tokens = fusionner_chiffres(tokens)
                    bboxes = [[m['left'], m['top'], m['left']+m['width'], m['top']+m['height']] for m in mots_pos]

                    start_idx = None
                    for i, tok in enumerate(tokens):
                        tok_norm = tok.replace(",", ".").replace(":", "").replace(" ", "").lower()
                        val_norm = valeur.replace(",", ".").replace(" ", "").lower()
                        if val_norm in tok_norm:
                            start_idx = i
                            break

                    if start_idx is None:
                        continue

                    qas.append({
                        "id": f"{filename}_{champ}",
                        "question": question,
                        "tokens": tokens,
                        "bboxes": bboxes,
                        "answers": [valeur],
                        "start_positions": [start_idx],
                        "end_positions": [start_idx]
                    })
                break
    numero_dossier = extraire_numero_dossier(lignes_corrigees)
    return qas, numero_dossier



def corriger_reponses(resultats, champs_a_corriger={"leucocytes", "plaquettes"}):
    # Ne rien changer aux valeurs, juste enlever les champs superflus
    resultats_simplifies = []
    for qa in resultats:
        resultats_simplifies.append({
            "id": qa["id"],
            "question": qa["question"],
            "tokens": qa["tokens"],
            "bboxes": qa["bboxes"]
        })
    return resultats_simplifies
def extraire_numero_dossier(lignes_corrigees):
    import re
    for ligne in lignes_corrigees:
        if "dossier n°" in ligne.lower():
            match = re.search(r"(\d+\s+\d+(?:\s*/\s*\d+)?)", ligne)
            if match:
                return match.group(1).strip()  
    return None
