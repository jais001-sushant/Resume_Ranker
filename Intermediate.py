import joblib
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import re
import os

# Paths to dataset and model files
DATASET_PATH = "/Users/jaiswal002sushant/Downloads/job_description_dataset.csv"
RESUME_DATASET_PATH = "/Users/jaiswal002sushant/Downloads/resume_ranker_dataset.csv"
MODEL_PATH = "resume_ranker_model.pkl"
VECTORIZER_PATH = 'vectorizer.pkl'

# Load the job description dataset
job_descriptions = pd.read_csv(DATASET_PATH)

# Train the model
def train_model():
    # Combine skills, courses, experience, and job description into a single input
    job_descriptions["Input"] = job_descriptions["Skills"] + " " + job_descriptions["Courses"] + " " + job_descriptions["Job Description"]
    
    # Vectorize job descriptions
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(job_descriptions["Input"])
    
    # Save the vectorizer for later use
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, job_descriptions["Eligibility %"])
    joblib.dump(model, MODEL_PATH)
    
    print("\nModel trained and saved.✅")

# Read resume PDF
def read_pdf_resume(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""  # Handle None if extract_text fails
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# Extract skills, experience, and courses from the resume
def extract_resume_info(resume_text):
    # Example extraction logic (you may need to adjust this based on your resume format)
    skills = re.findall(r'Skills:\s*(.*)', resume_text)
    experience = re.findall(r'Experience:\s*(\d+)', resume_text)
    courses = re.findall(r'Courses:\s*(.*)', resume_text)

    # Return the first found values or default to "Not found"
    return {
        "skills": skills[0] if skills else "Not found",
        "experience": experience[0] if experience else "Not Mentioned",
        "courses": courses[0] if courses else "Not found"
    }

# Predict eligibility score from resume
def predict_resume_score(resume_path):
    # Load the trained model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Read resume
    resume_text = read_pdf_resume(resume_path)

    # Extract skills, experience, and courses
    resume_info = extract_resume_info(resume_text)

    # Print extracted information
    print(f"\nSkills: {resume_info['skills']}")
    print(f"Experience: {resume_info['experience']} years")
    print(f"Courses: {resume_info['courses']}\n")

    # Prepare input for prediction
    resume_input = resume_info['skills'] + " " + resume_info['courses'] + " Experience " + resume_info['experience']
    
    # Vectorize resume input
    resume_vectorized = vectorizer.transform([resume_input])
    
    # Predict the eligibility score
    score = model.predict(resume_vectorized)[0]
    return round(score, 2)

# Main
if __name__ == "__main__":
    train = False  # Change to True to retrain the model, False to predict

    if train:
        train_model()
    else:
        resume_file = "/Users/jaiswal002sushant/Downloads/Sushant_Jaiswal_Resume_2025.pdf"
        if os.path.exists(resume_file):
            print(f"\nPredicting score for resume: {resume_file}")
            score = predict_resume_score(resume_file)
            
            print(f"Your predicted job eligibility score is: {score}%✨")
            
            if score >= 80:
                print("Excellent match! You're highly eligible.🎯")
            elif score >= 60:
                print("👍 Good match, but consider adding more skills.")
            else:
                print("⚠️ Needs improvement. Add relevant skills and experience.")
        else:
            print("⚠️ Resume file not found. Please check the path.")
