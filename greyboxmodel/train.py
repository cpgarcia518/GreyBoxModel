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
import copy
from timeit import default_timer as timer
from typing import Callable, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Own libraries
# ==============================================================================
# from darkgreybox import logger
from greyboxmodel.base_model import GreyModel

# Functions
def train_models(
    # models: Dict[str, GreyModel],
    models: List[GreyModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    error_metric: Callable,
    splits: Optional[List] = None,
    method: str = 'nelder',
    obj_func: Optional[Callable] = None,
    reduce_train_results: bool = False,
    n_jobs: int = -1,
    verbose: int = 10
) -> pd.DataFrame:
    """
    Trains the `models` for the given `X_train` and `y_train` training data
    for `splits` using `method`.

    Parameters
    ----------
    models: list of `model.GreyModel` objects
        list of models to be trained
    X_train: `pandas.DataFrame`
        A pandas DataFrame of the training input data X
    y_train: `pandas.Series`
        A pandas Series of the training input data y
    error_metric: Callable
        An error metric function that confirms to the `sklearn.metrics` interface
    splits: Optional[List]
        A list of training data indices specifying sub-sections of `X_train` and `y_train`
        for the models to be trained on
    method : str
        Name of the fitting method to use. Valid values are described in:
        `lmfit.minimize`
    obj_func: Optional[Callable]
        The objective function to minimise during the fitting
    n_jobs: Optional[int]
        The number of parallel jobs to be run as described by `joblib.Parallel`
    verbose: Optional[int]
        The degree of verbosity as described by `joblib.Parallel`

    Returns
    -------
    `pd.DataFrame` with a record for each model's result for each split

    Example:
    -------
    ```python

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
    method: str = 'nelder',
    obj_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Trains a copy of `base_model` for the given `X_train` and `y_train` training data
    using `method`.

    Parameters
    ----------
    base_model: `model.GreyModel`
        model to be trained (a copy will be made)
    X_train: `pandas.DataFrame`
        A pandas DataFrame of the training input data X
    y_train: `pandas.Series`
        A pandas Series of the training input data y
    error_metric: Callable
        An error metric function that confirms to the `sklearn.metrics` interface
    method : str
        Name of the fitting method to use. Valid values are described in:
        `lmfit.minimize`
    obj_func: Optional[Callable]
        The objective function to minimise during the fitting

    Returns
    -------
    `pd.DataFrame` with a single record for the fit model's result
    """

    start = timer()
    model = copy.deepcopy(base_model)

    start_date = X_train.index[0]
    end_date = X_train.index[-1]

    X = X_train.to_dict(orient='list')
    y = cast(np.ndarray, y_train.values)
    # y = y_train.to_numpy()

    try:
        model.fit(
            X=X,
            y=y,
            method=method,
            ic_params=get_ic_params(model, X_train),
            obj_func=obj_func
        )

    except ValueError:
        end = timer()
        return pd.DataFrame({
            'start_date': [start_date],
            'end_date': [end_date],
            'model': [np.NaN],
            'model_result': [np.NaN],
            'time': [end - start],
            'method': [method],
            'error': [np.NaN]
        })

    model_result = model.predict(X)
    end = timer()

    return pd.DataFrame({
        'start_date': [start_date],
        'end_date': [end_date],
        'model': [model],
        'model_result': [model_result],
        'time': [end - start],
        'method': [method],
        'error': [error_metric(y, model_result.Z)]
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

    # TODO: this is horrible - make this clearer and more robust
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

    return (
        df.replace([-np.inf, np.inf], np.nan)
        .dropna()
        .round({'error': decimals})
        .sort_values('time')
        .drop_duplicates(subset=['error'], keep='first')
        .sort_values('error')
        .reset_index(drop=True)
    )

if __name__ == "__main__":
    pass