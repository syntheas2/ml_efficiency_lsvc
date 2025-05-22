from pydantic import BaseModel
import torch

class Args(BaseModel):
    # MLflow
    mlflow_experiment_name: str = "ML_Efficiency_LinearSVC_Experiment"
    mlflow_run_name: str = "original_data_test"
