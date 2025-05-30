import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib
import os
import PyPDF2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Paths to dataset and model files
DATASET_PATH = "/Users/jaiswal002sushant/Downloads/resume_ranker_dataset.csv"   # Update with your dataset path
MODEL_PATH = "resume_ranker_model.pkl"      # Update with your model path

# Train model if not already trained
def train_model():
    df = pd.read_csv(DATASET_PATH)
    df.dropna(inplace=True)  # Drop missing values
    df["Input"] = df["Skills"] + " " + df["Courses"] + " Experience " + df["Experience (Years)"].astype(str)
    X = df["Input"]
    y = df["Label (Eligibility %)"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(), LinearRegression())
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nModel trained with MAE: {mae}, R²: {r2}")

    joblib.dump(model, MODEL_PATH)
    print("Model saved.✅\n")

# Read PDF resume and extract text
def read_pdf_resume(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Predict score from resume text
def predict_resume_score(resume_path):
    if not os.path.exists(MODEL_PATH):
        train_model()

    model = joblib.load(MODEL_PATH)

    if resume_path.endswith(".pdf"):
        resume_text = read_pdf_resume(resume_path)
    else:
        with open(resume_path, 'r', encoding='utf-8', errors='ignore') as f:
            resume_text = f.read()

    resume_input = resume_text + " Experience 0"
    score = model.predict([resume_input])[0]
    return round(score, 2)

# Main
if __name__ == "__main__":
    train = False  # Change to True to retrain the model, False to predict

    if train:
        train_model()
    else:
        resume_file = "/Users/jaiswal002sushant/Downloads/Sushant_Jaiswal_Resume_2025.pdf"  # Update with your resume path
        if os.path.exists(resume_file):
            score = predict_resume_score(resume_file)
            print(f"\nYour predicted job eligibility score is: {score}%")

            if score >= 80:
                print("🎯 Excellent match! You're highly eligible.")
            elif score >= 60:
                print("👍 Good match, but consider adding more skills.\n")
            else:
                print("⚠️ Needs improvement. Add relevant skills and experience.\n")
        else:
            print("⚠️ Resume file not found. Please check the path.\n")



