# 🧠 PromptBuddy Backend (FastAPI + ML Classifier)

This is the **backend API** for [PromptBuddy](https://github.com/robbype/PromptBuddy-Chrome-Extension) — a Chrome Extension that provides **real-time prompt feedback** inside ChatGPT.  
It uses a **Sentence Transformer** and a **Logistic Regression Multi-Label Classifier** to analyze prompts and return improvement suggestions with contextual hints.

---

## ✨ Features

- ⚡ **FastAPI** — lightweight and blazing fast backend framework.
- 🧠 **AI-powered classifier** — trained using Sentence Transformers.
- 💬 **Dynamic rule-based feedback** — customizable via `rules.json`.
- 🔗 **CORS-enabled API** — connect directly to your Chrome Extension.
- 💾 **Local model storage** — no cloud dependency required.

---

## ⚙️ Setup Instructions

### 1. Clone this repository
```bash
git clone https://github.com/robbype/PromptBuddy-Backend.git
cd PromptBuddy-Backend
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# or
.\.venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🧠 Train the Model

To (re)train your classifier, run:

```bash
python train.py
```
This will:

1. Load **`rules_dataset.csv`**  
2. Generate embeddings using **all-MiniLM-L6-v2**  
3. Train a multi-label classifier using **Logistic Regression**  
4. Save the following files:
   - `rule_classifier.pkl`
   - `label_encoder.pkl`
   - `embedding_model/`

## 🚀 Run the Server

Start the FastAPI server locally:
```bash
uvicorn main:app --reload
```

Your backend will be available at:
```bash
http://127.0.0.1:8000
```