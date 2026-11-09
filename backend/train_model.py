import json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================
# LOAD DATA
# ==========================
with open("../data/dataset_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🔥 Flatten nested lists
cleaned_data = []
for item in data:
    if isinstance(item, list):
        cleaned_data.extend(item)
    else:
        cleaned_data.append(item)

df = pd.DataFrame(cleaned_data).fillna("")

# ✅ FIX LIST → STRING ERROR
def normalize_text(value):
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value)

for col in df.columns:
    df[col] = df[col].apply(normalize_text)

# ==========================
# CREATE TEXT FIELD
# ==========================
df["text"] = (
    df["Category"] + " " +
    df["Subcategory"] + " " +
    df["Title/Name"] + " " +
    df["Description"] + " " +
    df["Extra Info"]
)

X_text = df["text"]
y = df["Category"]

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

# ==========================
# TF-IDF VECTORIZE
# ==========================
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================
# TRAIN MODEL
# ==========================
intent_model = LogisticRegression(max_iter=1000)
intent_model.fit(X_train_vec, y_train)

# ==========================
# PREDICTION
# ==========================
y_pred = intent_model.predict(X_test_vec)

# ==========================
# ACCURACY
# ==========================
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Accuracy:", round(accuracy * 100, 2), "%")

# ==========================
# MODEL COMPLEXITY
# ==========================
num_features = len(vectorizer.get_feature_names_out())
num_classes = len(intent_model.classes_)
num_coefficients = intent_model.coef_.size

print("\n🧠 Model Complexity Info")
print("Total Features:", num_features)
print("Number of Classes:", num_classes)
print("Total Coefficients:", num_coefficients)

# ==========================
# SAVE MODEL
# ==========================
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(intent_model, open("intent_model.pkl", "wb"))

print("\n💾 Model saved successfully")
