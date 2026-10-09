"""
Run the full training pipeline: ingest data -> transform -> train & save best model.

Usage:
    python -m src.pipeline.train_pipeline
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()

    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
        train_data_path, test_data_path
    )

    best_model_name, r2 = ModelTrainer().initiate_model_trainer(train_arr, test_arr)
    print(f"Training complete. Best model: {best_model_name} | Test R2: {r2:.4f}")
    return best_model_name, r2


if __name__ == "__main__":
    run_training_pipeline()
