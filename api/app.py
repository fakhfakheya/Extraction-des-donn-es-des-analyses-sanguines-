from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import tempfile, os, torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
import api_ocr
from correction import corriger_predictions
from insertion import inserer_predictions   # 👈 import ici

# Charger modèle LayoutLMv3 fine-tuné
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForQuestionAnswering.from_pretrained("./layoutlmv3_finetuned")
model.eval()

app = FastAPI(title="OCR + LayoutLMv3 API", version="1.0")

# --- Fonction utilitaire ---
def merge_tokens(tokens, bboxes):
    new_tokens, new_bboxes = [], []
    i = 0
    while i < len(tokens):
        if (
            i+1 < len(tokens)
            and tokens[i].replace(",", "").isdigit()
            and tokens[i+1].replace(",", "").isdigit()
        ):
            fused_token = tokens[i] + tokens[i+1]
            fused_bbox = [
                min(bboxes[i][0], bboxes[i+1][0]),
                min(bboxes[i][1], bboxes[i+1][1]),
                max(bboxes[i][2], bboxes[i+1][2]),
                max(bboxes[i][3], bboxes[i+1][3]),
            ]
            new_tokens.append(fused_token)
            new_bboxes.append(fused_bbox)
            i += 2
        else:
            new_tokens.append(tokens[i])
            new_bboxes.append(bboxes[i])
            i += 1
    return new_tokens, new_bboxes


# --- Route prédiction ---
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    ocr_results, numero_dossier = api_ocr.traiter_image(tmp_path, file.filename)
    predictions = []

    for example in ocr_results:
        image = Image.open(tmp_path).convert("RGB")
        width, height = image.size

        def normalize_bbox(bbox, w, h):
            return [
                int(1000 * (bbox[0] / w)),
                int(1000 * (bbox[1] / h)),
                int(1000 * (bbox[2] / w)),
                int(1000 * (bbox[3] / h)),
            ]

        bboxes = [normalize_bbox(b, width, height) for b in example["bboxes"]]
        tokens = example["tokens"]
        question = example["question"]

        tokens, bboxes = merge_tokens(tokens, bboxes)

        if len(tokens) != len(bboxes):
            print(f"⚠️ Skipping {example['id']} car tokens ({len(tokens)}) != bboxes ({len(bboxes)})")
            continue

        inputs = processor(
            image,
            text=[tokens],
            boxes=[bboxes],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)

        start_idx = torch.argmax(outputs.start_logits, dim=1).item()
        end_idx = torch.argmax(outputs.end_logits, dim=1).item()
        predicted_tokens = tokens[start_idx:end_idx+1]
        predicted_answer = " ".join(predicted_tokens)

        predictions.append({
            "question": question,
            "answer_predicted": predicted_answer,
        })

    os.remove(tmp_path)

    # 🔥 Correction
    predictions_corrigees = corriger_predictions(predictions, ocr_results)
    
    # ✅ Insertion + conversion (via insertion.py)
    return JSONResponse(content=inserer_predictions(predictions_corrigees, numero_dossier))


# --- Route test GET ---
@app.get("/")
async def root():
    return {"message": "API OCR + LayoutLMv3 fonctionne et MongoDB est connecté !"}
