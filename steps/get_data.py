import __init__ # noqa: F401
from scipy.sparse import csr_matrix
from typing import Tuple, Annotated
import pandas as pd
from zenml import step
from typing import Annotated, Any
from zenml.client import Client
import mlflow
from steps.get_x_y import get_x_y_test_step, get_x_y_train_step
from pythelpers.ml.mlflow_log import log_to_description
# from syntheas.data import get_df_test, get_df_train, get_df_val
from syntheas_mltools_mgmt.zenml.data import get_df_test_onehot, get_df_train_onehot, get_df_val_onehot, get_df_synthtrain_tabsyn_onehot3, get_df_synthtrain_tabsyn_onehot4

@step
def get_original_data_step() -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[csr_matrix, "X_test"],
    Annotated[pd.Series, "y_test"],
]:
    df_train = get_df_train_onehot()

    df_test = get_df_test_onehot()

    X_train, y_train, scaler = get_x_y_train_step(df_train)
    X_test, y_test = get_x_y_test_step(df_test, scaler)

    # Option 1: Log description as a tag (populates MLflow's run description/notes)
    run_description = (
        ("train with original data and test with original data\n\n"
        f"Training data shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}\n\n"
        f"Test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}\n\n"
        f"Learned minmax scaler on training data and applied it to test data\n\n")
    )
    mlflow.set_tag("mlflow.note.content", run_description)

    return X_train, y_train, X_test, y_test

@step
def get_original_data_with_val_step() -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[csr_matrix, "X_val"],
    Annotated[pd.Series, "y_val"],
]:
    df_train = get_df_train_onehot()

    df_val = get_df_val_onehot()

    X_train, y_train, scaler = get_x_y_train_step(df_train)
    X_val, y_val = get_x_y_test_step(df_val, scaler)

    # Option 1: Log description as a tag (populates MLflow's run description/notes)
    run_description = (
        ("train with original data and val with original data\n\n"
        f"Training data shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}\n\n"
        f"val data shape: {X_val.shape if hasattr(X_val, 'shape') else 'N/A'}\n\n"
        f"Learned minmax scaler on training data and applied it to val data\n\n")
    )
    mlflow.set_tag("mlflow.note.content", run_description)

    return X_train, y_train, X_val, y_val




@step
def get_test_data_step(train_cols, scaler) -> Tuple[
    Annotated[csr_matrix, "X_test"],
    Annotated[pd.Series, "y_test"],
]:
    df_test = get_df_test_onehot()
    df_test.drop(columns=['id', 'combined_tks'], inplace=True)
    df_test['impact'] = df_test['impact'].astype(str)


    df_test = df_test[train_cols]
    X_test, y_test = get_x_y_test_step(df_test, scaler)

    # Option 1: Log description as a tag (populates MLflow's run description/notes)
    log_to_description(
        ("test with orig data\n\n"
        f"Test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}\n\n"
        f"applied scaler to test data\n\n")
    )

    return X_test, y_test


@step
def get_cgan_data_step() -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[Any, "train_cols"],
    Annotated[Any, "scaler"],
]:
    artifact = Client().get_artifact_version("22886804-d443-4fe3-8a0a-fee83f2b537c")
    df_train = artifact.load()
    try:
        df_train.drop(columns=['id', 'combined_tks'], inplace=True)
    except Exception as e:
        print(f"Error dropping columns in train data: {e}")
    df_train['impact'] = df_train['impact'].astype(str)

    train_cols = df_train.columns
    X_train, y_train, scaler = get_x_y_train_step(df_train)

    # Option 1: Log description as a tag (populates MLflow's run description/notes)
    log_to_description(
        ("train with cgan data\n\n"
        f"Training data shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}\n\n"
        f"Learned minmax scaler on training data\n\n")
    )

    return X_train, y_train, train_cols, scaler

@step
def get_tabsyn_data_step() -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[csr_matrix, "X_test"],
    Annotated[pd.Series, "y_test"],
]:
    df_train = get_df_synthtrain_tabsyn_onehot3()
    try:
        df_train.drop(columns=['id', 'combined_tks'], inplace=True)
    except Exception as e:
        print(f"Error dropping columns in train data: {e}")
    df_train['impact'] = df_train['impact'].astype(str)


    df_test = get_df_test_onehot()
    df_test['impact'] = df_test['impact'].astype(str)

    df_test = df_test[df_train.columns]
    X_train, y_train, scaler = get_x_y_train_step(df_train)
    X_test, y_test = get_x_y_test_step(df_test, scaler)

    # Option 1: Log description as a tag (populates MLflow's run description/notes)
    log_to_description(
        ("train with tabsyn data and test with original data\n\n"
        f"Training data shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}\n\n"
        f"Test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}\n\n"
        f"Learned minmax scaler on training data and applied it to test data\n\n")
    )

    return X_train, y_train, X_test, y_test

@step
def get_tabsyn_data2_step() -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[csr_matrix, "X_test"],
    Annotated[pd.Series, "y_test"],
]:
    df_train = get_df_synthtrain_tabsyn_onehot3()
    try:
        df_train.drop(columns=['id', 'combined_tks'], inplace=True)
    except Exception as e:
        print(f"Error dropping columns in train data: {e}")
    df_train['impact'] = df_train['impact'].astype(str)

    df_train3 = get_df_synthtrain_tabsyn_onehot4()
    try:
        df_train3.drop(columns=['id', 'combined_tks'], inplace=True)
    except Exception as e:
        print(f"Error dropping columns in train data: {e}")
    df_train3['impact'] = df_train3['impact'].astype(str)

    df_train2 = get_df_train_onehot()
    try:
        df_train2.drop(columns=['id', 'combined_tks'], inplace=True)
    except Exception as e:
        print(f"Error dropping columns in train data: {e}")
    df_train2['impact'] = df_train2['impact'].astype(str)
    
    df_train_combined = pd.concat([df_train, df_train2, df_train3], ignore_index=True)

    df_test = get_df_test_onehot()
    df_test['impact'] = df_test['impact'].astype(str)

    df_test = df_test[df_train_combined.columns]
    X_train, y_train, scaler = get_x_y_train_step(df_train_combined)
    X_test, y_test = get_x_y_test_step(df_test, scaler)

    # Option 1: Log description as a tag (populates MLflow's run description/notes)
    log_to_description(
        ("train with tabsyn data mixed with orig and test with original data\n\n"
        f"Training data shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}\n\n"
        f"Test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}\n\n"
        f"Learned minmax scaler on training data and applied it to test data\n\n")
    )


    return X_train, y_train, X_test, y_test

@step
def get_tabsyn_data3_step() -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[csr_matrix, "X_test"],
    Annotated[pd.Series, "y_test"],
]:
    df_train = get_df_synthtrain_tabsyn_onehot3()
    try:
        df_train.drop(columns=['id', 'combined_tks'], inplace=True)
    except Exception as e:
        print(f"Error dropping columns in train data: {e}")
    df_train['impact'] = df_train['impact'].astype(str)

    df_train2 = get_df_synthtrain_tabsyn_onehot4()
    try:
        df_train2.drop(columns=['id', 'combined_tks'], inplace=True)
    except Exception as e:
        print(f"Error dropping columns in train data: {e}")
    df_train2['impact'] = df_train2['impact'].astype(str)
    
    df_train_combined = pd.concat([df_train, df_train2], ignore_index=True)

    df_test = get_df_test_onehot()
    df_test['impact'] = df_test['impact'].astype(str)

    df_test = df_test[df_train_combined.columns]
    X_train, y_train, scaler = get_x_y_train_step(df_train_combined)
    X_test, y_test = get_x_y_test_step(df_test, scaler)

    # Option 1: Log description as a tag (populates MLflow's run description/notes)
    log_to_description(
        ("train with tabsyn data mixed with orig and test with original data\n\n"
        f"Training data shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}\n\n"
        f"Test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}\n\n"
        f"Learned minmax scaler on training data and applied it to test data\n\n")
    )


    return X_train, y_train, X_test, y_test
