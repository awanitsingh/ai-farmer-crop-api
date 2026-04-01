"""
Crop Recommendation Model - Improved Training Script
Dataset: Crop_recommendation.csv (included in repo)
Run: python3 train_crop_model.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import kagglehub
import os

# ── Download dataset ──────────────────────────────────────────────────────────
print("Downloading crop recommendation dataset...")
try:
    path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
    csv_path = os.path.join(path, "Crop_recommendation.csv")
except Exception:
    csv_path = "./dataset.csv"  # fallback to local

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")
print(df['label'].value_counts())

# ── Prepare data ──────────────────────────────────────────────────────────────
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values
le = LabelEncoder()
y = le.fit_transform(df['label'])

print(f"\nClasses: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Train ensemble model ──────────────────────────────────────────────────────
print("\nTraining ensemble model...")

rf  = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)

ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# ── Save model and label encoder ─────────────────────────────────────────────
with open("classifier.pkl", "wb") as f:
    pickle.dump(ensemble, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Saved classifier.pkl and label_encoder.pkl")
print("Training complete!")
