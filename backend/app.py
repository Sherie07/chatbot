from bert_model import prepare_bert, get_bert_response
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

app = Flask(__name__)
CORS(app)

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "../data/dataset_cleaned.json")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ===============================
# LOAD DATA + FLATTEN JSON
# ===============================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def flatten_json(data):
    flat_list = []
    if isinstance(data, list):
        for item in data:
            flat_list.extend(flatten_json(item))
    elif isinstance(data, dict):
        flat_list.append(data)
    return flat_list

data = flatten_json(data)
df = pd.DataFrame(data)

# ===============================
# ENSURE REQUIRED COLUMNS EXIST
# ===============================
required_columns = [
    "Category",
    "Subcategory",
    "Title/Name",
    "Description",
    "Class",
    "Aliases",
    "Extra Info",
    "Eligible Courses",
    "Level"
]

for col in required_columns:
    if col not in df.columns:
        df[col] = ""

# ===============================
# NORMALIZE LIST VALUES
# ===============================
def normalize_value(val):
    if isinstance(val, list):
        return ", ".join(map(str, val))
    return str(val)

for col in df.columns:
    df[col] = df[col].apply(normalize_value)

# ===============================
# CREATE SEARCH TEXT
# ===============================
df["text"] = (
    df["Category"].astype(str) + " " +
    df["Subcategory"].astype(str) + " " +
    df["Title/Name"].astype(str) + " " +
    df["Description"].astype(str) + " " +
    df["Class"].astype(str) + " " +
    df["Aliases"].astype(str)
)
prepare_bert(df)
# ===============================
# LOAD ML MODELS
# ===============================
try:
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    X = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    print("❌ Model loading failed:", e)
    vectorizer = None
    X = None

# ===============================
# SYLLABUS FILE MAP
# ===============================
syllabus_map = {
    "bca": "syllabus/bca.pdf",
    "msc": "syllabus/msc_cs.pdf",
    "mca": "syllabus/mca_cs.pdf",
    "bsc_aiml1": "syllabus/bscaiml_1.pdf",
    "bsc_aiml2": "syllabus/bscaiml2.pdf",
    "pg_diploma": "syllabus/pgdds.pdf"
}

# ===============================
# CLEAN TEXT
# ===============================
def clean_text(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

# ===============================
# INTENT DETECTION
# ===============================
def detect_intent(text):
    text = text.lower()

    if "can i" in text or "eligible" in text:
     return "Eligibility"

    if "document" in text or "documents" in text:
        return "Documents"

    if any(w in text for w in ["eligibility", "apply", "admission", "qualification"]):
        return "Eligibility"

    if any(w in text for w in ["fee", "fees", "structure", "cost"]):
        return "Fees"

    if any(w in text for w in ["course", "subject", "syllabus"]):
        return "Course"

    if any(w in text for w in ["faculty", "professor", "teacher"]):
        return "Faculty"

    if "scholarship" in text:
        return "Scholarship"

    if any(w in text for w in ["schedule", "timetable", "lecture", "time"]):
        return "Schedule"

    return None

# ===============================
# LEVEL DETECTION
# ===============================
def detect_level(text):
    text = text.lower()

    if "ug" in text or "bsc" in text or "bca" in text:
        return "UG"

    if "pgdds" in text:
        return "PGDDS"

    if "pg" in text or "msc" in text or "mca" in text:
        return "PG"

    return None

# ===============================
# COURSE DETECTION
# ===============================
def detect_course(text):

    cleaned = clean_text(text)

    if "bscaiml1" in cleaned or "aiml1" in cleaned:
        return "bsc_aiml1"

    if "bscaiml2" in cleaned or "aiml2" in cleaned:
        return "bsc_aiml2"

    if "bscaiml" in cleaned or "aiml" in cleaned:
        return "bsc_aiml1"

    courses = {
        "bsc": ["bsc", "bachelor"],
        "msc": ["msc", "master"],
        "phd": ["phd"],
        "mca": ["mca"],
        "bca": ["bca"]
    }

    for course, keywords in courses.items():
        for key in keywords:
            if key in cleaned:
                return course

    return None

# ===============================
# COURSE SYNONYM MAPPER (NEW)
# ===============================
course_synonyms = {
    "bsc_aiml1": ["bsc ai ml", "bsc aiml", "aiml", "ai ml"],
    "bsc_aiml2": ["bsc ai ml", "bsc aiml", "aiml", "ai ml"],
    "pg_diploma": ["pgdds", "pg diploma", "data science diploma"],
    "msc": ["msc", "computer science"],
    "mca": ["mca"],
    "bca": ["bca"]
}

def match_course_in_row(row_text, course):

    if not course:
        return True

    synonyms = course_synonyms.get(course, [course])
    row_text = str(row_text).lower()

    for syn in synonyms:
        if syn in row_text:
            return True

    return False

# ===============================
# EXTRACT DAYS
# ===============================
def extract_days(description):
    description = description.lower()

    day_map = {
        "mon": "Monday",
        "tue": "Tuesday",
        "wed": "Wednesday",
        "thu": "Thursday",
        "fri": "Friday",
        "sat": "Saturday"
    }

    days = []
    for short, full in day_map.items():
        if short in description:
            days.append(full)

    return days

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return "University Chatbot Running 🚀"

# ===============================
# DOWNLOAD SYLLABUS
# ===============================
@app.route("/download_syllabus/<course>")
def download_syllabus(course):

    file_path = syllabus_map.get(course)

    if not file_path:
        return "File not found", 404

    abs_path = os.path.join(BASE_DIR, file_path)

    if os.path.exists(abs_path):
        return send_file(abs_path, as_attachment=True)

    return "File not found", 404

# ===============================
# CHAT ROUTE
# ===============================
@app.route("/chat", methods=["POST"])
def chat():

    user_input = request.json.get("message", "")
    user_lower = user_input.lower().strip()

    if user_lower in ["hi", "hello", "hey"]:
        return jsonify({"reply": "Hello 😊 How can I help you?"})

    if user_lower in ["thanks", "thank you"]:
        return jsonify({"reply": "You're welcome 🙌"})

    intent = detect_intent(user_lower)
    course = detect_course(user_lower)
    level = detect_level(user_lower)

    # ===============================
    # ELIGIBILITY FIX (NEW)
    # ===============================
    if intent == "Eligibility":

        elig_df = df[
            df["Category"].str.contains("Eligibility", case=False, na=False)
        ]

        if course:
            elig_df = elig_df[
                elig_df.apply(
                    lambda r: match_course_in_row(
                        str(r["Title/Name"]) +
                        str(r["Aliases"]) +
                        str(r["Eligible Courses"]),
                        course
                    ),
                    axis=1
                )
            ]

        if elig_df.empty:
            return jsonify({"reply": "❌ No eligibility info found."})

        row = elig_df.iloc[0]

        reply = f"📌 {row['Description']}"

        if pd.notna(row.get("Extra Info")) and row.get("Extra Info"):
            reply += f"<br><br><b>Extra Info:</b> {row['Extra Info']}"

        return jsonify({"reply": reply})

    # ===============================
    # ADMISSION DOCUMENTS
    # ===============================
    if intent == "Documents":

        doc_df = df[
            (df["Category"].str.contains("Admission", case=False, na=False)) &
            (df["Subcategory"].str.contains("Documents", case=False, na=False))
        ]

        if level:
            doc_df = doc_df[
                doc_df["Level"].str.contains(level, case=False, na=False)
            ]

        if doc_df.empty:
            return jsonify({"reply": "❌ No document list found."})

        reply = "📄 <b>Required Admission Documents</b><br><br>"

        for _, row in doc_df.iterrows():
            docs = row["Description"].split(",")
            reply += f"<b>{row['Title/Name']}</b><br>"

            for d in docs:
                reply += f"• {d.strip()}<br>"

            reply += "<br>"

        return jsonify({"reply": reply})

    # ===============================
    # SYLLABUS
    # ===============================
    if intent == "Course" and course:

        file_path = syllabus_map.get(course)

        if file_path:
            abs_path = os.path.join(BASE_DIR, file_path)

            if os.path.exists(abs_path):

                pretty_course = course.replace("_", " ").upper()

                return jsonify({
                    "reply": f"📄 Here’s the syllabus for {pretty_course}",
                    "file": f"/download_syllabus/{course}"
                })

        return jsonify({"reply": "❌ Syllabus not found for that course."})

    # ===============================
    # SCHOLARSHIP
    # ===============================
    if intent == "Scholarship":

        scholarship_df = df[
            df["Category"].str.contains("Scholarship", case=False, na=False)
        ]

        if course:
            scholarship_df = scholarship_df[
                scholarship_df.apply(
                    lambda r: match_course_in_row(
                        str(r["Eligible Courses"]) + str(r["Title/Name"]),
                        course
                    ),
                    axis=1
                )
            ]

        if scholarship_df.empty:
            return jsonify({"reply": "❌ No scholarship found for this course."})

        reply = f"🎓 <b>Scholarships for {course.upper() if course else ''}</b><br><br>"

        for _, row in scholarship_df.iterrows():
            reply += f"""
📌 <b>{row['Title/Name']}</b><br>
{row['Description']}<br>
Eligibility: {row.get('Eligible Courses','')}<br><br>
"""

        return jsonify({"reply": reply})

    # ===============================
    # TIMETABLE
    # ===============================
    if intent == "Schedule":

        timetable_df = df[df["Category"].str.contains("Schedule", case=False, na=False)]

        if course:
            timetable_df = timetable_df[
                timetable_df["Class"].astype(str).str.lower().str.contains(course, na=False)
            ]

        if timetable_df.empty:
            return jsonify({"reply": "❌ No timetable found."})

        day_schedule = {d: [] for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]}

        for _, row in timetable_df.iterrows():

            days = extract_days(str(row["Description"]))

            for day in days:
                lecture = f"""
📌 <b>{row['Title/Name']}</b><br>
🕒 {row['Description']}<br>
🏫 {row.get('Extra Info', '')}<br><br>
"""
                day_schedule[day].append(lecture)

        pretty_course = course.replace("_", " ").upper() if course else ""
        reply = f"📅 <b>{pretty_course} Weekly Timetable</b><br><br>"

        for day, lectures in day_schedule.items():
            if lectures:
                reply += f"<b>{day}:</b><br>" + "".join(lectures)

        return jsonify({"reply": reply})

    # ===============================
    # NORMAL ML MATCH
    # ===============================
    filtered_df = df.copy()

    if intent:
        filtered_df = filtered_df[
            filtered_df["Category"].str.contains(intent, case=False, na=False)
        ]

    if course:
        filtered_df = filtered_df[
            filtered_df.apply(
                lambda r: match_course_in_row(r["text"], course),
                axis=1
            )
        ]

    if filtered_df.empty:
        return jsonify({"reply": "🤔 Could not find matching info."})

    if vectorizer is None:
        return jsonify({"reply": "❌ ML model not loaded."})

        # ===============================
        # BERT FALLBACK (NEW)
        # ===============================
        bert_reply = get_bert_response(user_input, df)

        if bert_reply:
            return jsonify({"reply": bert_reply})
    filtered_vec = vectorizer.transform(filtered_df["text"])
    user_vec = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vec, filtered_vec)
    index = similarity.argmax()

    row = filtered_df.iloc[index]

    reply = f"📌 {row['Description']}"

    if pd.notna(row.get("Extra Info")) and row.get("Extra Info"):
        reply += f"<br><br><b>Extra Info:</b> {row['Extra Info']}"

    return jsonify({"reply": reply})

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
