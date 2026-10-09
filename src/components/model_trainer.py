import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Optional boosted-tree libraries: fall back gracefully if not installed
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }
            params = {
                "Decision Tree": {"max_depth": [3, 5, 7, 10, None]},
                "Random Forest Regressor": {"n_estimators": [50, 100, 200]},
                "AdaBoost Regressor": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.5]},
            }

            if _HAS_CATBOOST:
                models["CatBoosting Regressor"] = CatBoostRegressor(verbose=False, random_state=42)
            if _HAS_XGBOOST:
                models["XGBRegressor"] = XGBRegressor(random_state=42)

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, params=params,
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No sufficiently good model found (best R2 < 0.6)", sys)

            logging.info(f"Best model: {best_model_name} with R2 score {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return best_model_name, r2

        except Exception as e:
            raise CustomException(e, sys)
