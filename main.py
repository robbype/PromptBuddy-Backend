from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import json
import re

app = FastAPI(title="Rule Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("📦 Memuat model dan classifier...")

model = SentenceTransformer("embedding_model")

clf = joblib.load("rule_classifier.pkl")
mlb = joblib.load("label_encoder.pkl")

PROBA_THRESHOLD = 0.5

class InputText(BaseModel):
    text: str

@app.post("/analyze")
def analyze(data: InputText):
    text = data.text

    emb = model.encode([text])

    try:
        proba = clf.predict_proba(emb)

        if isinstance(proba, list):
            pred = np.array([ (p[0] if p.shape[0]==1 else p) for p in proba ]).T  
        else:
            pred = proba 

        pred_bin = (pred[0] >= PROBA_THRESHOLD).astype(int)
        labels = mlb.inverse_transform([pred_bin])
    except AttributeError:
        pred = clf.predict(emb)
        labels = mlb.inverse_transform(pred)

    theLabel = list(labels[0])
    prompt = ""
    hint = ""

    with open("rules.json") as f:
        feedback_rules = json.load(f)

    if not theLabel:
        sentences = re.split(r'[.!?]\s+', text.strip())
        sentence_count = len([s for s in sentences if s]) 

        if sentence_count < 10:
            prompt = f"Teks Anda terlalu singkat ({sentence_count} kalimat). " \
                     "Berikan konteks yang lebih mendalam agar konversi ke bahasa awam lebih akurat."
            hint = "Coba tambahkan detail teknis atau contoh kasus pada teks Anda."
        else:
            prompt = "Teks sudah cukup panjang dan tidak terdeteksi pelanggaran aturan."
            hint = "Anda bisa melanjutkan ke proses berikutnya."

    elif len(theLabel) == 1:
        label = theLabel[0]
        rule_data = feedback_rules["rules"].get(label, {})
        prompt = rule_data.get("prompt", "")
        hint = rule_data.get("hint", "")
    else:
        combo_key = "+".join(sorted(theLabel))
        combo_data = feedback_rules.get("combos", {}).get(combo_key)
        if combo_data:
            prompt = combo_data.get("prompt")
            hint = combo_data.get("hint")

    return {
        "text": text,
        "rules": labels[0] if labels else [],
        "feedback": prompt if prompt else "",
        "hints": hint
    }
