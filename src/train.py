import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


MODELS = {
    "xgboost": XGBRegressor(n_estimators=500, learning_rate=0.05),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=300),
}


def train_with_tracking(X_train, y_train, preprocessor):
    """Train multiple models with MLflow tracking.

    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Fitted preprocessor from features.py

    Returns:
        Best trained pipeline based on cross-validation RMSE
    """
    best_model, best_score = None, float("inf")

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                [("preprocessor", preprocessor), ("model", model)]
            )

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=5,
                scoring="neg_root_mean_squared_error",
            )
            rmse = -scores.mean()

            mlflow.log_param("model", name)
            mlflow.log_metric("cv_rmse", rmse)
            mlflow.log_metric("cv_std", scores.std())

            pipeline.fit(X_train, y_train)
            mlflow.sklearn.log_model(pipeline, "model")

            print(f"{name}: RMSE={rmse:.4f} ± {scores.std():.4f}")

            if rmse < best_score:
                best_score, best_model = rmse, pipeline

    return best_model
