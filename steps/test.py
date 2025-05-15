import __init__ # noqa: F401
from typing import Annotated
from zenml import step
from sklearn.svm import LinearSVC
import sklearn
import mlflow # Import the mlflow library


@step
def test_step(model: LinearSVC, X_test, y_test) -> Annotated[LinearSVC, "model"]:
    """
    Evaluates the model on test data and logs metrics to MLflow with a 'test_' prefix.
    """

    # Evaluierung des Modells -> Berechnung von Verhersagen an Testdaten
    prediction = model.predict(X_test)

    # Berechnung von Evaluierungsmetriken
    report_dict = sklearn.metrics.classification_report(y_test, prediction, output_dict=True)

    # --- Explicit MLflow Logging for the classification report with 'test_' prefix ---

    # 1. Log the entire classification report as a artifact
    mlflow.log_text(sklearn.metrics.classification_report(y_test, prediction), "test_classification_report.txt")

    # 2. Log key metrics individually for easier tracking and comparison in MLflow UI
    # Log overall accuracy
    if "accuracy" in report_dict:
        mlflow.log_metric("test_accuracy", report_dict["accuracy"])

    # Log metrics for each class (precision, recall, f1-score)
    for class_label, metrics in report_dict.items():
        if isinstance(metrics, dict): # Check if the item is a dictionary of metrics
            if "precision" in metrics:
                mlflow.log_metric(f"test_{class_label}_precision", metrics["precision"])
            if "recall" in metrics:
                mlflow.log_metric(f"test_{class_label}_recall", metrics["recall"])
            if "f1-score" in metrics:
                mlflow.log_metric(f"test_{class_label}_f1-score", metrics["f1-score"])
            if "support" in metrics:
                mlflow.log_metric(f"test_{class_label}_support", metrics["support"])

    # Log macro and weighted averages for precision, recall, f1-score
    for avg_type in ["macro avg", "weighted avg"]:
        if avg_type in report_dict:
            avg_metrics = report_dict[avg_type]
            if "precision" in avg_metrics:
                mlflow.log_metric(f"test_{avg_type.replace(' ', '_')}_precision", avg_metrics["precision"])
            if "recall" in avg_metrics:
                mlflow.log_metric(f"test_{avg_type.replace(' ', '_')}_recall", avg_metrics["recall"])
            if "f1-score" in avg_metrics:
                mlflow.log_metric(f"test_{avg_type.replace(' ', '_')}_f1-score", avg_metrics["f1-score"])

    # The model itself is passed through.
    return model
