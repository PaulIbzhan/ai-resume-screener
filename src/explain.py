import shap # type: ignore
import pandas as pd
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore

def extract_keywords_from_jd(jd_text, top_k=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=top_k)
    X = vectorizer.fit_transform([jd_text])
    return vectorizer.get_feature_names_out()

def extract_features(text, keywords):
    text = text.lower()
    return [int(keyword in text) for keyword in keywords]

def get_shap_values(resume_text, job_description):
    # Step 1: Extract dynamic keywords from JD
    keywords = extract_keywords_from_jd(job_description)

    # Step 2: Extract features from resume & JD
    resume_features = extract_features(resume_text, keywords)
    jd_features = extract_features(job_description, keywords)

    # Step 3: Create DataFrames
    X = pd.DataFrame([resume_features], columns=keywords)
    training_data = pd.DataFrame([
        jd_features,
        [0] * len(keywords)
    ], columns=keywords)

    # Step 4: Simulated labels and model training
    y = [1, 0]

    model = LogisticRegression()
    model.fit(training_data, y)

    # Step 5: SHAP Explanation
    explainer = shap.Explainer(model, training_data)
    shap_values = explainer(X)

    return X, shap_values
