import __init__ # noqa: F401
from zenml import pipeline
from steps.get_data import get_tabsyn_data3_step
from steps.train import train_step
from steps.test import test_step
import mlflow  # Import the mlflow library
from datetime import datetime # For the run description
from pydantic import BaseModel

class Args(BaseModel):
    # MLflow
    mlflow_experiment_name: str = "ML_Efficiency_LinearSVC"
    mlflow_run_name: str = "tabsyn_500k_data_test"

@pipeline
def train_tabsyn_data_with_test_pipeline():
    args = Args()
    
    # --- Set the MLflow run name using a tag ---
    now = datetime.now() # Use current time for uniqueness
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.set_tag("mlflow.runName", f"{args.mlflow_run_name}_{timestamp_str}")
    
    X_train, y_train, X_test, y_test = get_tabsyn_data3_step()
    model = train_step(X_train, y_train)
    model = test_step(model, X_test, y_test)

    return model

if __name__ == "__main__":
    train_tabsyn_data_with_test_pipeline.with_options(
        enable_cache=False  
    )()