import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib

print("🔹 Reading dataset...")
df = pd.read_csv("rules_dataset.csv")
df["labels"] = df["labels"].apply(lambda x: x.split(","))

print("🔹 Converting text into embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode(df["text"].tolist(), show_progress_bar=True)

print("🔹 Creating binary labels...")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["labels"])

print("🔹 Training classifier...")
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf.fit(X, y)

# Save the model and label encoder
joblib.dump(clf, "rule_classifier.pkl")
joblib.dump(mlb, "label_encoder.pkl")
model.save("embedding_model")

print("✅ Training complete! Model and encoder have been saved.")
