import __init__ # noqa: F401
from scipy.sparse import csr_matrix
from typing import Tuple, Annotated
import pandas as pd
from zenml import step
from typing import Annotated
import pandas as pd
import pandas as pd
from scipy.sparse import hstack
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy


@step
def get_x_y_train_step(df_train) -> Tuple[
    Annotated[csr_matrix, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[MinMaxScaler, "scaler"],
]:
    # Separation der Zielvariable
    y = df_train['impact']

    # separation von numerischen Attributen und Festsetzung auf float Vektor
    df_numeric = df_train.select_dtypes(include=['number', 'bool'])#.drop(columns= 'impact')
    X_vec1 = np.array(df_numeric).astype(float)
    
    scaler = MinMaxScaler()
    X_vec1 = scaler.fit_transform(X_vec1)

    # Horizonatle Verbindung des Numerik-Vektors
    X_combined = hstack([scipy.sparse.csr_matrix(X_vec1)])
    
    return X_combined, y, scaler


@step
def get_x_y_test_step(df_test, scaler) -> Tuple[
    Annotated[csr_matrix, "X_test"],
    Annotated[pd.Series, "y_test"],
]:
    # Separation der Zielvariable
    y = df_test['impact']

    # separation von numerischen Attributen und Festsetzung auf float Vektor
    df_numeric = df_test.select_dtypes(include=['number', 'bool'])#.drop(columns= 'impact')
    X_vec1 = np.array(df_numeric).astype(float)
    
    X_vec1 = scaler.transform(X_vec1)

    # Horizonatle Verbindung des Numerik-Vektors
    X_combined = hstack([scipy.sparse.csr_matrix(X_vec1)])
    
    return X_combined, y
