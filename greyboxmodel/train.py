#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
""" Train the grey box model """

from __future__ import annotations

__author__ = "Carlos Alejandro Perez Garcia"
__copyright__ = "Copyright 2023"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Carlos Alejandro Perez Garcia"
__email__ = "cpgarcia518@gmail.com"

# Standard libraries
# ==============================================================================
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from copy import deepcopy
from timeit import default_timer as timer

from typing import Dict, Optional, Callable, List, cast

# Own libraries
# ==============================================================================
from greyboxmodel.base_model import GreyModel

def train_models(
    # models: Dict[str, GreyModel],
    models: List[GreyModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    error_metric: Callable,
    splits: Optional[List] = None,
    method: str = "nelder",
    obj_func: Optional[Callable] = None,
    reduce_train_results: bool = True,
    n_jobs: int = 1,
    verbose: int = 10
) -> pd.DataFrame:
    """ Trains the `models` for the given `X_train` and `y_train` training data for `splits` using `method`.

    Parameters
    ----------
    models: Dict[str, GreyModel]
        Dictionary of models to train.
    X_train: pd.DataFrame
        Training data.
    y_train: pd.Series
        A pandas Series of the training input data y.
    error_metric: Callable
        Error metric function that confirms to the `sklearn.metrics` interface.
    splits: Optional[List]
        A list of training data indices specifying sub-sections of `X_train` and `y_train`
        for the models to be trained on
    method: str
        Optimization method to use. Default is `nelder`. Valid values are described in:
            `lmfit.minimize`
    obj_func: Optional[Callable]
        Objective function to minimize. If None, the default objective function of the
        base model will be used.
    reduce_train_results: bool
        Whether to reduce the training results dataframe by removing nan and duplicate records.
    n_jobs: int
        Number of jobs to run in parallel as described by `joblib.Parallel`.
    verbose: int
        The verbosity level as described by `joblib.Parallel`.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the training results.

    Example:
    ```

    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    from darkgreybox.model import TiTe
    from darkgreybox.fit import train_models


    prefit_df = train_models(
        models=[TiTe(train_params, rec_duration=1)],
        X_train=X_train,
        y_train=y_train,
        splits=KFold(n_splits=int(len(X_train) / 24), shuffle=False).split(X_train),
        error_metric=mean_squared_error,
        method='nelder',
        n_jobs=-1,
        verbose=10
    )
    ```
    """
    # TODO: Add type hints into __init__ for logger
    # logger.info('Training models...')

    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs, verbose=verbose) as p:
            df = cast(pd.DataFrame, pd.concat(
                cast(pd. DataFrame, p(delayed(train_model)(
                    model, X_train.iloc[idx], y_train.iloc[idx], error_metric, method, obj_func)
                    for _, idx in splits or [(None, range(len(X_train)))] for model in models
                )),
                ignore_index=True
            ))

    else:
        df = cast(pd.DataFrame, pd.concat([
            train_model(model, cast(pd.DataFrame, X_train.iloc[idx]), y_train.iloc[idx], error_metric, method, obj_func)
            for _, idx in splits or [(None, range(len(X_train)))] for model in models
        ],
            ignore_index=True))

    if reduce_train_results:
        return reduce_results_df(df)
    else:
        return df


def train_model(
    base_model: GreyModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    error_metric: Callable,
    method: str = "nelder",
    obj_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Train a copy of the base model with the given data and method.

    Parameters
    ----------
    base_model: GreyModel
        Base model to train (a copy will be made).
    X_train: pd.DataFrame
        Training data.
    y_train: pd.Series
        A pandas Series of the training input data y.
    error_metric: Callable
        Error metric function that confirms to the `sklearn.metrics` interface.
    method: str
        Optimization method to use. Default is `nelder`. Valid values are described in:
            `lmfit.minimize`
    obj_func: Optional[Callable]
        Objective function to minimize. If None, the default objective function of the
        base model will be used.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the training results.
    """

    start = timer()
    model = deepcopy(base_model)

    start_date = X_train.index[0]
    end_date = X_train.index[-1]

    X = X_train.to_dict(orient='list')
    y = y_train.to_numpy()
    # y = cast(np.ndarray, y_train.values)

    # Fit the model
    # result = model.fit(X, y, method=method, objective_func=obj_func)
    try:
        model.fit(X, y, method=method, ic_params=get_ic_params(model, X_train), obj_func=obj_func)
    except ValueError:
        end = timer()
        return pd.DataFrame({
            'start_date': [start_date],
            'end_date': [end_date],
            'model': [np.NaN],
            'model_result': [np.NaN],
            'time': [end - start],
            'method': [method],
            'error_metric': [np.NaN]
            })

    # Get the model predictions
    model_result = model.predict(X)
    end = timer()

    return pd.DataFrame({
        'start_date': [start_date],
        'end_date': [end_date],
        'model': [model],
        'model_result': [model_result],
        'time': [end - start],
        'method': [method],
        # 'error': [error_metric(y, model_result.Z)]
        'error_metric': [error_metric(y_train, model_result.yhat)],
    })

def get_ic_params(model: GreyModel, X_train: pd.DataFrame) -> Dict:
    """
    Get the initial conditions parameters for the model.

    Parameters
    ----------
    model: `model.GreyModel`
        Model to get initial condition parameters from
    X_train: `pd.DataFrame`
        Training data

    Returns
    -------
    Dict
        A dictionary of initial condition parameters.
    """

    ic_params = {}
    for key in model.params:
        if '0' in key:
            # TODO: This is solution could be better
            # ic_params[key] = model.params[key].value

            if key in X_train:
                ic_params[key] = X_train.iloc[0][key]
            else:
                raise KeyError(f'Initial condition key {key} does not have corresponding X_train field')

    return ic_params

def reduce_results_df(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """Reduce `df` dataframe by removing nan and duplicate records

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to reduce/clean
    decimals : int, optional
        The number of decimal points for the float comparison when removing duplicates, by default 6

    Returns
    -------
    pd.DataFrame
        Reduced dataframe
    """

    return(
        df.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .round({'error': decimals})
        .sort_values(by=['time'])
        .drop_duplicates(subset=['error'], keep='first')
        .sort_values(by=['error'])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    pass