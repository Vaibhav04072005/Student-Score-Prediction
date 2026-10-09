# Student Performance Indicator

Predicts a student's **math score** from demographic info and other exam
scores (reading score, writing score). Deployed as a **Streamlit** app
(no Flask needed).

## Project structure

```
mlproject/
├── artifacts/                  # generated at training time (model.pkl, preprocessor.pkl, train/test/raw csv)
├── data/
│   └── stud.csv                 # sample data (synthetic placeholder — swap in the real Kaggle dataset)
├── notebook/
│   ├── 1_EDA_STUDENT_PERFORMANCE.ipynb
│   ├── 2_MODEL_TRAINING.ipynb
│   └── data/stud.csv             # same data, used by the training pipeline
├── src/
│   ├── exception.py               # custom exception with file/line info
│   ├── logger.py                  # logging setup (writes to ./logs)
│   ├── utils.py                   # save/load objects, evaluate_models()
│   ├── components/
│   │   ├── data_ingestion.py      # reads CSV, splits train/test, saves to artifacts/
│   │   ├── data_transformation.py # OneHotEncoder + StandardScaler pipeline
│   │   └── model_trainer.py       # trains & compares several regressors, saves the best
│   └── pipeline/
│       ├── train_pipeline.py      # orchestrates ingestion → transformation → training
│       └── predict_pipeline.py    # loads artifacts and predicts on new input
├── streamlit_app.py               # the app (replaces app.py / templates/home.html)
├── requirements.txt
├── setup.py
└── .gitignore
```

## ⚠️ About the data

`data/stud.csv` and `notebook/data/stud.csv` shipped here are **synthetic**
placeholder data with the same 8 columns as the real dataset, so the
pipeline runs end-to-end out of the box. For real results, replace both
files with the actual dataset:
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model (CLI, optional)

```bash
python -m src.pipeline.train_pipeline
```

This creates `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

## Run the app

```bash
streamlit run streamlit_app.py
```

If no trained model is found yet, use the **"Train / Retrain model"**
button in the sidebar — it runs the same pipeline as above, then the
prediction form becomes usable.

## Pipeline summary

- **Features**: gender, race/ethnicity, parental level of education, lunch
  type, test preparation course, reading score, writing score.
- **Target**: math score.
- **Preprocessing**: median/most-frequent imputation, `OneHotEncoder` on
  categorical columns, `StandardScaler` on numeric columns.
- **Models compared**: Linear Regression, Ridge, Lasso, KNN, Decision Tree,
  Random Forest, AdaBoost, CatBoost, XGBoost (last two only if installed).
  The best model by test R² is saved and used for inference.
