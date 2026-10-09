import os
import sys

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """Trains each model (optionally tuned via RandomizedSearchCV when params
    are provided) and returns a dict of {model_name: test_r2_score}."""
    try:
        report = {}

        for name, model in models.items():
            param_grid = params.get(name, {})

            if param_grid:
                rs = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42)
                rs.fit(X_train, y_train)
                model.set_params(**rs.best_params_)

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)

            report[name] = test_r2

        return report

    except Exception as e:
        raise CustomException(e, sys)
