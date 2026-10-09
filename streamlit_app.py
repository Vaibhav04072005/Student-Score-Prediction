"""
Student Performance Indicator - Streamlit App
Predicts a student's Math Score. Replaces the original Flask app.

Run with:
    streamlit run streamlit_app.py
"""

import os
import numpy as np
import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import run_training_pipeline

st.set_page_config(page_title="Student Performance Indicator", page_icon="🎓", layout="centered")

MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

CAT_OPTIONS = {
    "gender": ["male", "female"],
    "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
    "parental_level_of_education": [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree",
    ],
    "lunch": ["standard", "free/reduced"],
    "test_preparation_course": ["none", "completed"],
}


def main():
    st.title("🎓 Student Performance Indicator")
    st.write("Predict a student's **Math Score** from their background and other exam scores.")

    artifacts_ready = os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)

    with st.sidebar:
        st.header("Model")
        if artifacts_ready:
            st.success("Trained model found in artifacts/")
        else:
            st.warning("No trained model found yet.")
        if st.button("Train / Retrain model"):
            with st.spinner("Running training pipeline (data ingestion → transformation → training)..."):
                best_model_name, r2 = run_training_pipeline()
            st.success(f"Trained **{best_model_name}** — test R² = {r2:.4f}")
            st.rerun()

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        st.info(
            "Click **Train / Retrain model** in the sidebar to run the pipeline on "
            "`notebook/data/stud.csv` and generate `artifacts/model.pkl` + "
            "`artifacts/preprocessor.pkl`."
        )
        st.stop()

    st.subheader("Enter student details")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", CAT_OPTIONS["gender"])
        race_ethnicity = st.selectbox("Race/Ethnicity", CAT_OPTIONS["race_ethnicity"])
        parental_level_of_education = st.selectbox(
            "Parental Level of Education", CAT_OPTIONS["parental_level_of_education"]
        )
        lunch = st.selectbox("Lunch Type", CAT_OPTIONS["lunch"])
    with col2:
        test_preparation_course = st.selectbox("Test Preparation Course", CAT_OPTIONS["test_preparation_course"])
        reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
        writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

    if st.button("Predict Math Score", type="primary"):
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )
        pred_df = data.get_data_as_data_frame()

        prediction = PredictPipeline().predict(pred_df)[0]
        prediction = float(np.clip(prediction, 0, 100))
        st.metric("Predicted Math Score", f"{prediction:.1f} / 100")


if __name__ == "__main__":
    main()
