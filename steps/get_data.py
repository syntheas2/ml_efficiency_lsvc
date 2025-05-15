import __init__ # noqa: F401
from scipy.sparse import csr_matrix
from typing import Tuple, Annotated
import pandas as pd
from zenml import step
from typing import Annotated
from zenml.client import Client
import mlflow
from steps.get_x_y import get_x_y_test_step, get_x_y_train_step


@step
def get_original_data_step() -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[csr_matrix, "X_test"],
    Annotated[pd.Series, "y_test"],
]:
    artifact = Client().get_artifact_version(
        "cf513ab8-4c45-4d8a-b049-1f5d1932a3b4")
    df_train = artifact.load()

    artifact2 = Client().get_artifact_version(
        "8ff20b28-5e55-4541-a558-81d2d351cc72")
    df_test = artifact2.load()

    X_train, y_train, scaler = get_x_y_train_step(df_train)
    X_test, y_test = get_x_y_test_step(df_test, scaler)

    # Option 1: Log description as a tag (populates MLflow's run description/notes)
    run_description = (
        "train with original data and test with original data\n\n"
        f"Training data shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}\n\n"
        f"Test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}\n\n"
        f"Learned minmax scaler on training data and applied it to test data\n\n"
    )
    mlflow.set_tag("mlflow.note.content", run_description)

    return X_train, y_train, X_test, y_test
