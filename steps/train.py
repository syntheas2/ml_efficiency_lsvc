import __init__ # noqa: F401
from typing import Annotated
from zenml import step
from sklearn.svm import LinearSVC
from src.model import get_model
import mlflow # Import the mlflow library



@step
def train_step(X_train, y_train) -> Annotated[LinearSVC, "model"]:
    # Enable MLflow autologging for scikit-learn
    # This will automatically log parameters, metrics, and the model
    mlflow.sklearn.autolog()

    model = get_model()

    # Modelltraining
    model.fit(X_train, y_train)

    return model
